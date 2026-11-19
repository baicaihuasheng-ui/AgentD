"""Local encoder/AgentD acceptance. DeepPK requires explicit --deeppk."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deeppk", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import pandas as pd
    import requests
    import torch
    from agentD.affinity import AffinityPredictor, atomic_json
    from agentD.runtime_config import load_runtime
    from agentD.tools import prediction

    source = args.output.resolve() / "example-molecules.csv"
    examples = pd.DataFrame({"ID": ["aspirin", "caffeine"],
                             "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]})
    if not source.exists():
        examples.to_csv(source, index=False)
    prediction.PROPERTY_DIR = str(args.output.resolve())
    report_path = args.output / "acceptance.json"
    report = {"status": "running", "torch": torch.__version__, "cuda": torch.cuda.is_available(), "checks": {}}
    if args.deeppk:
        report = json.loads(report_path.read_text()) if report_path.exists() else report
        result = prediction.get_admet_predictions.invoke({"csv_file_path": str(source)})
        report["deeppk"] = {"result": result, "passed": isinstance(result, str) and result.startswith("ADMET predictions saved")}
        report["status"] = "passed" if report["deeppk"]["passed"] and not report.get("error") else "failed"
        atomic_json(report_path, report)
        return 0 if report["deeppk"]["passed"] else 1

    def blocked(*args, **kwargs):
        raise AssertionError("Network access blocked during affinity acceptance")
    requests.sessions.Session.request = blocked
    socket.create_connection = blocked
    socket.socket.connect = blocked
    report["network_blocked"] = True
    config = load_runtime()
    try:
        short = AffinityPredictor(config)
        # First sequence from the pinned ProtT5 model card.
        report["checks"]["official_short"] = {"sequence": "PRTEINO", "values": short.predict("PRTEINO", examples.SMILES.tolist()), "metrics": short.report()}
        atomic_json(report_path, report)
        fasta = (ROOT / "reports/fixtures/P10415.fasta").read_text()
        assert "BCL2_HUMAN" in fasta and "Homo sapiens" in fasta
        sequence = "".join(fasta.splitlines()[1:])
        report["protein"] = {"accession": "P10415", "source": "https://rest.uniprot.org/uniprotkb/P10415.fasta",
                             "length": len(sequence), "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}
        first = AffinityPredictor(config)
        values = first.predict(sequence, examples.SMILES.tolist())
        report["checks"]["bcl2_first"] = {"values": values, "metrics": first.report()}
        atomic_json(report_path, report)
        # Remove just this sequence's cache by using a separate directory for a cold repeat.
        repeat = AffinityPredictor(replace(config, affinity_cache_path=args.output.resolve() / f"repeat-cache-{time.time_ns()}"))
        repeated = repeat.predict(sequence, examples.SMILES.tolist())
        assert not repeat.cache_hit
        delta = max(abs(a-b) for a, b in zip(values, repeated))
        report["checks"]["cold_repeat"] = {"values": repeated, "max_abs_difference": delta, "metrics": repeat.report()}
        assert delta <= 1e-5, f"Repeated affinity differs by {delta}"
        result = prediction.predict_affinity_batch.invoke({"data": json.dumps({"sequence": sequence, "smiles_path": str(source)})})
        if not isinstance(result, tuple):
            raise RuntimeError(str(result))
        frame = pd.read_csv(result[1])
        assert frame.ID.tolist() == examples.ID.tolist()
        assert max(abs(a-b) for a,b in zip(values, frame["Affinity [pKd]"])) <= 1e-5
        metrics = json.loads(Path(result[1]).with_suffix(".metrics.json").read_text())
        assert metrics["cache_hit"]
        report["checks"]["agentd_tool"] = {"path": result[1], "metrics": metrics, "passed": True}
        report["status"] = "passed"
        atomic_json(report_path, report)
        print(json.dumps({"status": "passed", "report": str(report_path), "values": values, "repeat_delta": delta}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    sys.exit(main())
