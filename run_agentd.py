#!/usr/bin/env python
"""
AgentD Pipeline Runner - Drug Discovery Automation

Usage:
    # Run with default config (BCL-2 / Venetoclax test case)
    conda run -n agentd python run_agentd.py
    
    # Run with custom YAML config
    conda run -n agentd python run_agentd.py --config my_config.yaml
    
    # Q&A mode (RAG-based) - requires terminal with stdin support
    conda activate agentd && python run_agentd.py --qna
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Suppress RDKit C-level stderr warnings at the earliest possible point
import warnings
warnings.filterwarnings("ignore")

# Prevent local mcp.py from shadowing the mcp SDK
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sys_path_backup = sys.path[:]
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

try:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
finally:
    sys.path = _sys_path_backup

from agentD.runtime_config import configure_runtime,load_runtime


# Default configuration
DEFAULT_CONFIG = {
    "protein": "BCL-2",
    "disease": "chronic lymphocytic leukemia",
    "iterations": 2,
    "num_smiles": 2,  
    "run_boltz": True,
    "boltz_top_k": 2,
    "boltz_extra_args_json": '["--use_msa_server","--accelerator","gpu","--num_workers","4"]',
    "model": "deepseek-flash",
    "min_qed": 0.50,   # Minimum QED score for candidate selection
    "min_pkd": 5.0,    # Minimum pKd value for candidate selection
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    config["model"] = load_runtime().model
    
    if config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        if user_config is not None and not isinstance(user_config, dict):
            raise ValueError("Pipeline config must be a YAML mapping")
        if user_config:
            config.update(user_config)
        print(f"Loaded config from: {config_path}")
    
    return config


def find_latest_run_dir():
    """Find the most recent run directory."""
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(runs_dir):
        return None
    dirs = [d for d in os.listdir(runs_dir) if os.path.isfile(os.path.join(runs_dir, d, "status.json"))]
    if not dirs:
        return None
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d, "status.json")), reverse=True)
    return os.path.join(runs_dir, dirs[0])


async def monitor_progress_background(stop_event: asyncio.Event, run_dir: Path, interval: float = 5):
    """Track the status file belonging to this pipeline invocation."""
    run_dir = Path(run_dir).resolve()
    status_file = run_dir / "status.json"
    last_display = None
    last_error = None
    print(f"\n  [RUN DIR] {run_dir}", flush=True)

    while True:
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            current_step = status.get("current_step", "unknown")
            steps = status.get("steps", [])
            details = steps[-1].get("details", {}) if steps else {}
            display = (current_step, json.dumps(details, sort_keys=True, ensure_ascii=False))
            if display != last_display:
                detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
                timestamp = time.strftime("%H:%M:%S")
                suffix = f" | {detail_text}" if detail_text else ""
                print(f"  [{timestamp}] {current_step}{suffix}", flush=True)
                last_display = display
            last_error = None
        except (FileNotFoundError, json.JSONDecodeError):
            if last_display is None:
                print("  等待初始化：status.json 尚未就绪", flush=True)
                last_display = ("waiting", "")
        except Exception as error:
            diagnostic = f"{type(error).__name__}: {error}"
            if diagnostic != last_error:
                print(f"  进度读取失败 [{status_file}] {diagnostic}", flush=True)
                last_error = diagnostic

        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_qna_mode(config: dict):
    """Run interactive Q&A mode via MCP."""
    server_script = os.path.join(os.path.dirname(__file__), "mcp_agent.py")
    transport = StdioTransport(
        command=sys.executable,
        args=[server_script],
        cwd=os.path.dirname(server_script),
        env={
            **{k: v for k, v in os.environ.items() if k.startswith(("AGENTD_", "DEEPSEEK_", "HF_", "TRANSFORMERS_"))},
            "AGENTD_FORCE_CONDA_RUN": "0",
            "PYTHONNOUSERSITE": "1",
            "TMP": os.environ.get("TMP", os.environ.get("TEMP", "")),
        },
    )

    async with Client(transport) as client:
        print("=" * 70)
        print("AgentD Q&A Mode (RAG-based Research Paper Q&A)")
        print("=" * 70)
        print("\nThis mode downloads relevant papers and answers questions using RAG.")
        print("Type 'quit' or 'exit' to leave.\n")

        default_protein = config.get("protein", "")
        default_disease = config.get("disease", "")
        default_model = config.get("model", "deepseek-flash")

        print(f"Default protein: {default_protein or '(none)'}")
        print(f"Default disease: {default_disease or '(none)'}")
        print()

        protein_input = input(f"Enter protein name [{default_protein}]: ").strip()
        protein = protein_input if protein_input else default_protein or None

        disease_input = input(f"Enter disease name [{default_disease}]: ").strip()
        disease = disease_input if disease_input else default_disease or None

        qna_run_id = None
        
        # Download papers immediately if protein and disease are provided
        if protein and disease:
            print(f"\nDownloading papers for {protein} + {disease}...")
            start_time = time.time()
            
            # Call agentd_qna with a dummy question to trigger paper download
            result = await client.call_tool("agentd_qna", {
                "question": "_init_papers_",  # Special init command
                "protein": protein,
                "disease": disease,
                "model": default_model,
            })
            
            elapsed = time.time() - start_time
            
            if result and hasattr(result, 'content'):
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        try:
                            data = json.loads(content_item.text)
                            qna_run_id = data.get("run_id")
                            papers = data.get("papers_downloaded", [])
                            print(f"Downloaded {len(papers)} papers to runs/{qna_run_id}/papers/ ({elapsed:.1f}s)")
                        except json.JSONDecodeError:
                            pass
            
            print("\nReady for questions!")
        
        print("-" * 70)

        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    if qna_run_id:
                        print(f"Q&A session saved to: runs/{qna_run_id}/")
                    break
                if not question:
                    continue

                print("\nQuerying AgentD Q&A...")
                start_time = time.time()

                # Build params - always pass run_id to reuse same session
                params = {
                    "question": question,
                    "model": default_model,
                }
                
                # Pass run_id to reuse existing session (papers already downloaded)
                if qna_run_id:
                    params["run_id"] = qna_run_id

                result = await client.call_tool("agentd_qna", params)

                elapsed = time.time() - start_time

                print("\n" + "=" * 50)
                if result and hasattr(result, 'content'):
                    for content_item in result.content:
                        if hasattr(content_item, 'text'):
                            try:
                                data = json.loads(content_item.text)
                                
                                # Capture run_id from first response if not set
                                if not qna_run_id and data.get("run_id"):
                                    qna_run_id = data["run_id"]
                                    print(f"[Session: {qna_run_id}]")
                                
                                if data.get("ok"):
                                    print(f"Answer: {data.get('answer', 'No answer')}")
                                else:
                                    print(f"Error: {data.get('error', 'Unknown error')}")
                            except json.JSONDecodeError:
                                print(content_item.text)
                print(f"(Time: {elapsed:.1f}s)")
                print("=" * 50)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


async def run_pipeline_mode(config: dict):
    """Run full pipeline mode with given config."""
    config = dict(config)
    run_id = config.get("run_id") or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    config["run_id"] = run_id
    run_dir = (Path(__file__).resolve().parent / "runs" / run_id).resolve()
    server_log = run_dir / "logs" / "mcp_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [MCP LOG] {server_log}", flush=True)
    server_script = os.path.join(os.path.dirname(__file__), "mcp_agent.py")
    transport = StdioTransport(
        log_file=server_log,
        keep_alive=False,
        command=sys.executable,
        args=[server_script],
        cwd=os.path.dirname(server_script),
        env={
            **{k: v for k, v in os.environ.items() if k.startswith(("AGENTD_", "DEEPSEEK_", "HF_", "TRANSFORMERS_"))},
            "AGENTD_FORCE_CONDA_RUN": "0",
            "PYTHONNOUSERSITE": "1",
            "TMP": os.environ.get("TMP", os.environ.get("TEMP", "")),
        },
    )

    async with Client(transport) as client:
        print("=" * 70)
        print("AgentD Drug Discovery Pipeline")
        print("=" * 70)

        # List available tools
        tools = await client.list_tools()
        print(f"\nAvailable tools: {[t.name for t in tools]}\n")

        # Display configuration
        print("Configuration:")
        if config.get("run_id"):
            print(f"  Run ID: {config['run_id']}")
        print(f"  Protein: {config['protein']}")
        print(f"  Disease: {config['disease']}")
        print(f"  (Drug will be discovered via LLM extraction)")
        print(f"  Iterations: {config['iterations']}")
        print(f"  Num SMILES: {config['num_smiles']} per model")
        print(f"  Run Boltz: {config['run_boltz']}")
        print(f"  Boltz Top K: {config['boltz_top_k']}")
        print(f"  Min QED: {config.get('min_qed', 0.50)}")
        print(f"  Min pKd: {config.get('min_pkd', 5.0)}")
        print(f"  Model: {config['model']}")
        print("-" * 70)
        print("\nProgress updates will appear below:")
        
        start_time = time.time()
        
        # Start background progress monitor
        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_progress_background(stop_monitor, run_dir))
    
        # Build pipeline parameters
        pipeline_params = {
            "protein": config["protein"],
            "disease": config["disease"],
            "iterations": config["iterations"],
            "num_smiles": config["num_smiles"],
            "run_boltz": config["run_boltz"],
            "boltz_top_k": config["boltz_top_k"],
            "boltz_extra_args_json": config["boltz_extra_args_json"],
            "model": config["model"],
            "min_qed": config.get("min_qed", 0.50),
            "min_pkd": config.get("min_pkd", 5.0),
        }
        
        # Add run_id if specified in config
        if config.get("run_id"):
            pipeline_params["run_id"] = config["run_id"]
        
        try:
            result = await client.call_tool("agentd_run_pipeline", pipeline_params)
        finally:
            stop_monitor.set()
            await monitor_task
        
        elapsed = time.time() - start_time

        # Ensure stdout is not suppressed
        sys.stdout.flush()
        
        print("\n" + "=" * 70, flush=True)
        print(f"Pipeline Result (elapsed: {elapsed:.1f}s / {elapsed/60:.1f}min):", flush=True)
        print("=" * 70, flush=True)
        
        # Parse and pretty-print the result
        pipeline_ok = False
        if result and hasattr(result, 'content'):
            for content_item in result.content:
                if hasattr(content_item, 'text'):
                    try:
                        data = json.loads(content_item.text)
                        print(json.dumps(data, indent=2), flush=True)
                        
                        # Print summary
                        if isinstance(data, dict):
                            pipeline_ok = data.get("ok") is True and not getattr(result,"is_error",False)
                            print("\n" + "-" * 70, flush=True)
                            print("SUMMARY:", flush=True)
                            print(f"  Run ID: {data.get('run_id', 'N/A')}", flush=True)
                            print(f"  Run Dir: {data.get('run_dir', 'N/A')}", flush=True)
                            print(f"  OK: {data.get('ok', 'N/A')}", flush=True)
                            if 'error' in data:
                                print(f"  ERROR: {data['error']}", flush=True)
                            if 'iterations' in data:
                                print(f"  Iterations completed: {len(data['iterations'])}", flush=True)
                                for it in data['iterations']:
                                    is_last = it.get('is_last', False)
                                    if is_last:
                                        print(f"    - iter_{it['iter']:02d}: ok={it.get('ok')}, is_last=True", flush=True)
                                        print(f"      Property files: affinity_before, admet_before, affinity_after, admet_after", flush=True)
                                    else:
                                        print(f"    - iter_{it['iter']:02d}: ok={it.get('ok')}", flush=True)
                            if 'boltz' in data:
                                print(f"  Boltz structures generated: {len(data['boltz'].get('yamls', []))}", flush=True)
                                print(f"  Boltz jobs: {len(data['boltz'].get('jobs', []))}", flush=True)
                    except json.JSONDecodeError:
                        print(content_item.text, flush=True)
        else:
            print(result, flush=True)

        print("\n" + "=" * 70, flush=True)
        print("PIPELINE COMPLETED SUCCESSFULLY!" if pipeline_ok else "PIPELINE FAILED — inspect result and logs", flush=True)
        print("=" * 70, flush=True)
        print("\nCheck status.json for detailed pipeline status.", flush=True)
        print(flush=True)
        
        return pipeline_ok




async def main():
    parser = argparse.ArgumentParser(
        description="AgentD Drug Discovery Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config
  python run_agentd.py --config pipeline_config.yaml

  # Q&A mode
  python run_agentd.py --qna --config pipeline_config.yaml
        """
    )
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--qna", action="store_true", help="Run in Q&A mode (RAG-based)")
    args = parser.parse_args()

    # Load config (used by both pipeline and Q&A modes)
    config = load_config(args.config)
    configure_runtime(config)

    if args.qna:
        await run_qna_mode(config)
    else:
        return 0 if await run_pipeline_mode(config) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
