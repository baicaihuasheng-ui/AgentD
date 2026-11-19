from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from agentD.runtime_config import load_runtime, configure_runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_gpu_measurement_initializes_allocator():
    from agentD.affinity import measure_stage
    metrics = []
    with measure_stage(metrics, "test", "cuda:0"):
        value = torch.ones(4, device="cuda:0")
        assert value.sum().item() == 4
    assert metrics[0]["peak_cuda_allocated_bytes"] > 0


def test_affinity_settings_reach_subprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTD_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setenv("AGENTD_PROTEIN_MODEL_PATH", "before-test")
    monkeypatch.setenv("AGENTD_AFFINITY_DEVICE", "auto")
    configure_runtime({"protein_model_path": str(tmp_path), "affinity_device": "cpu"})
    config = load_runtime()
    assert config.protein_model_path == tmp_path
    assert config.affinity_device == "cpu"


def test_sequential_extraction_and_cpu_head(monkeypatch, tmp_path):
    from agentD.affinity import AffinityPredictor
    calls = []
    config = replace(load_runtime(), affinity_cache_path=tmp_path, affinity_device="cpu")
    predictor = AffinityPredictor(config)
    monkeypatch.setattr(predictor, "_protein", lambda sequence, device: (calls.append("protein") or torch.ones(1, 1024)))
    monkeypatch.setattr(predictor, "_molecules", lambda smiles: (calls.append("molecules") or torch.ones(len(smiles), 768)))
    result = predictor.predict("UZOB", ["CCO", "CC"])
    assert calls == ["protein", "molecules"]
    assert len(result) == 2 and all(torch.isfinite(torch.tensor(result)))
    assert predictor.protein_sequence == "X X X X"
    assert predictor.protein_features.dtype == torch.float32


def test_invalid_smiles_rejected_before_loading_models():
    from agentD.affinity import AffinityPredictor
    with pytest.raises(ValueError, match="SMILES"):
        AffinityPredictor().predict("ACDE", ["not-a-molecule"])


def test_oom_retries_protein_once_on_cpu(monkeypatch, tmp_path):
    from agentD.affinity import AffinityPredictor
    predictor = AffinityPredictor(replace(load_runtime(), affinity_device="cuda:0", affinity_cache_path=tmp_path))
    calls = []
    def protein(sequence, device):
        calls.append(device)
        if device.startswith("cuda"):
            raise torch.cuda.OutOfMemoryError("test")
        return torch.ones(1, 1024)
    monkeypatch.setattr(predictor, "_protein", protein)
    monkeypatch.setattr(predictor, "_molecules", lambda smiles: torch.ones(len(smiles), 768))
    predictor.predict("ACDE", ["CCO"])
    assert calls == ["cuda:0", "cpu"]


def test_affinity_tool_preserves_rows_and_filename(monkeypatch, tmp_path):
    from agentD.tools import prediction
    source = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["a", "b"], "SMILES": ["CCO", "CC"]}).to_csv(source, index=False)
    monkeypatch.setattr(prediction, "PROPERTY_DIR", str(tmp_path / "output"))
    monkeypatch.setattr("agentD.affinity.AffinityPredictor.predict", lambda *args: [1.2, 3.4])
    monkeypatch.setattr("agentD.affinity.AffinityPredictor.report", lambda *args: {"test": True})
    result = prediction.predict_affinity_batch.invoke({"data": json.dumps({"sequence": "ACDE", "smiles_path": str(source)})})
    output = Path(result[1])
    assert output.name == "affinity_input.csv"
    frame = pd.read_csv(output)
    assert frame.ID.tolist() == ["a", "b"]
    assert frame["Affinity [pKd]"].tolist() == [1.2, 3.4]


def test_missing_protein_model_fails_without_network(tmp_path):
    from agentD.affinity import AffinityPredictor
    config = replace(load_runtime(), protein_model_path=tmp_path / "missing", affinity_cache_path=tmp_path / "cache", affinity_device="cpu")
    with pytest.raises(OSError):
        AffinityPredictor(config).predict("ACDE", ["CCO"])


def test_workers_propagate_prediction_failure(monkeypatch, tmp_path):
    import mcp_agent
    from agentD.tools import prediction
    from unittest.mock import Mock
    monkeypatch.chdir(tmp_path)
    affinity = Mock(return_value="Error predicting affinity")
    affinity.invoke.return_value = "Error predicting affinity"
    admet = Mock(return_value={"error": "test"})
    admet.invoke.return_value = {"error": "test"}
    monkeypatch.setattr(prediction, "predict_affinity_batch", affinity)
    monkeypatch.setattr(prediction, "get_admet_predictions", admet)
    assert mcp_agent._worker_predict_affinity(sequence="ACDE", smiles_csv="missing.csv", out_dir=str(tmp_path)) != 0
    assert mcp_agent._worker_predict_admet(smiles_csv="missing.csv", out_dir=str(tmp_path)) != 0
