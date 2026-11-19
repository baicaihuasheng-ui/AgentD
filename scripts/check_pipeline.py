"""Explicit, one-batch launcher with a shared budget and process-tree timeout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def terminate_tree(process, known_children=()):
    import psutil
    targets = list(known_children)
    try:
        parent = psutil.Process(process.pid)
        targets.extend(parent.children(recursive=True))
        targets.append(parent)
    except psutil.NoSuchProcess:
        pass
    for target in reversed(targets):
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(targets, timeout=10)


def record_interrupted_candidates(run_dir, error):
    from mcp_agent import _failed_iteration
    for iteration in Path(run_dir).glob("iter_*"):
        source = iteration / "pool/candidates.csv"
        if source.is_file() and not (iteration / "property/comparison.csv").exists():
            _failed_iteration(source, iteration, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirmed-budget-rmb", type=float)
    parser.add_argument("--resume-run", help="Recover an interrupted pooling stage using its existing budget and deadline")
    args = parser.parse_args()
    from agentD.runtime_config import load_runtime
    config = load_runtime()
    if not args.live:
        print(json.dumps({"key_configured": bool(config.api_key), "model": config.model,
                          "chat_limit": 20, "search_limit": 6, "fetch_limit": 3, "max_rmb": 1.0,
                          "status": "configuration_only"}))
        return 0
    if args.confirmed_budget_rmb != 1.0:
        parser.error("This batch requires explicit --confirmed-budget-rmb 1 after fee confirmation")
    config.require_key()
    if config.model != "deepseek-flash" or config.api_base != "https://api.deepseek.com":
        raise ValueError("This approved batch requires the official deepseek-flash endpoint")
    from agentD.call_budget import CallBudget
    from agentD.affinity import atomic_json
    run_id = args.resume_run or "BCL2_loop_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run ID")
    run_dir = ROOT / "runs" / run_id
    budget_path = run_dir / "usage.json"
    attempts = []
    if args.resume_run:
        state = json.loads((run_dir / "status.json").read_text())
        if state["current_step"] not in {"pooling", "iteration_0_prediction"}:
            raise ValueError("Only a pooling/baseline-prediction checkpoint can be recovered")
        CallBudget(budget_path).check_deadline()
        previous = run_dir / "acceptance.json"
        if previous.exists():
            prior = json.loads(previous.read_text())
            attempts = prior.pop("previous_attempts", []) + [prior]
        if not (run_dir / "extraction.json").is_file():
            raise ValueError("Missing extraction checkpoint")
    else:
        run_dir.mkdir(parents=True)
        CallBudget.create(budget_path)
    import yaml
    task = yaml.safe_load((ROOT / "pipeline_config.deepseek.yaml").read_text())
    task.update(run_id=run_id, iterations=1, num_smiles=2, run_boltz=False, boltz_top_k=0)
    task_path = run_dir / "task.yaml"
    task_path.write_text(yaml.safe_dump(task), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(AGENTD_BUDGET_PATH=str(budget_path), AGENTD_FORCE_CONDA_RUN="0")
    if args.resume_run:
        environment["AGENTD_RESUME_POOLING"] = "1"
    started = time.time()
    with (run_dir / "runner.log").open("ab" if args.resume_run else "wb") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "run_agentd.py"), "--config", str(task_path)],
                                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        import psutil
        children = {}
        try:
            parent = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    for child in parent.children(recursive=True):
                        children[child.pid] = child
                except psutil.NoSuchProcess:
                    pass
                if time.time() >= json.loads(budget_path.read_text())["deadline"]:
                    raise subprocess.TimeoutExpired(process.args, 1800)
                time.sleep(0.5)
            code = process.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            code = 124
        finally:
            terminate_tree(process, children.values())
    result = {"run_id": run_id, "run_dir": str(run_dir), "returncode": code, "elapsed_seconds": time.time()-started,
              "status": "passed" if code == 0 else "failed", "usage": str(budget_path), "previous_attempts": attempts}
    result["batch_elapsed_seconds"] = time.time() - (json.loads(budget_path.read_text())["deadline"] - 1800)
    if code:
        record_interrupted_candidates(run_dir, "Run deadline exceeded" if code == 124 else "Pipeline interrupted or failed")
    atomic_json(run_dir / "acceptance.json", result)
    print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
