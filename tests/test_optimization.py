import pandas as pd
import pytest
from types import SimpleNamespace


def test_merge_by_candidate_id_and_validate_molecule():
    from agentD.optimization import merge_predictions
    candidates = pd.DataFrame({"candidate_id": ["c1", "c2"], "SMILES": ["CCO", "CC"]})
    affinity = candidates.copy()
    affinity["Affinity [pKd]"] = [1.0, 2.0]
    admet = candidates.iloc[::-1].copy()
    admet["[Example] Predictions"] = [3, 4]
    result = merge_predictions(candidates, affinity, admet)
    assert result.candidate_id.tolist() == ["c1", "c2"]
    assert result["[Example] Predictions"].tolist() == [4, 3]
    with pytest.raises(ValueError):
        merge_predictions(candidates, affinity, admet.iloc[:1])
    with pytest.raises(ValueError):
        merge_predictions(candidates, affinity, pd.concat([admet, admet.iloc[:1]]))
    affinity.loc[0, "Affinity [pKd]"] = float("nan")
    with pytest.raises(ValueError):
        merge_predictions(candidates, affinity, admet)


def test_refinement_keeps_errors_unchanged_and_confidence(tmp_path):
    from agentD.utils import process_dataset_with_agent
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "SMILES": ["CCO"] * 4,
                          "Affinity [pKd]": [1.2] * 4, "[Example] Probability": [0.8] * 4})
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    replies = iter(["CCN</s>Affinity [pKd]</s>test", "OCC</s>Affinity [pKd]</s>unchanged", "bad format", "not-smiles</s>x</s>test"])
    prompts = []
    def invoke(data):
        prompts.append(data["input"])
        return {"output": next(replies)}
    result = process_dataset_with_agent(str(source), "BCL-2", [], SimpleNamespace(invoke=invoke))
    assert result.candidate_id.tolist() == ["a", "b", "c", "d"]
    assert result.status.tolist() == ["modified", "unchanged", "failed", "failed"]
    assert "Probability" in prompts[0] and "1.2" in prompts[0]


def test_comparison_reuses_unchanged_and_keeps_failed_rows():
    from agentD.optimization import comparison_table
    before = pd.DataFrame({"candidate_id": ["a", "b", "c"], "SMILES": ["CCO"] * 3,
                           "Affinity [pKd]": [1.0] * 3, "[Example] Predictions": [2.0] * 3})
    refined = pd.DataFrame({"candidate_id": ["a", "b", "c"], "SMILES": ["CCO"] * 3,
                            "Updated_SMILES": ["CCN", "OCC", ""], "status": ["modified", "unchanged", "failed"],
                            "Property": ["Affinity [pKd]"] * 3, "Rationale": ["test"] * 3, "error": ["", "", "invalid"]})
    after = pd.DataFrame({"candidate_id": ["a"], "SMILES": ["CCN"], "Affinity [pKd]": [3.0], "[Example] Predictions": [4.0]})
    result = comparison_table(before, refined, after)
    assert result["delta_Affinity [pKd]"].iloc[0] == 2.0
    assert result.prediction_source.tolist() == ["predicted", "reused", "unavailable"]
    assert pd.isna(result["after_Affinity [pKd]"].iloc[2])
    assert "before_QED" in result and "after_MW" in result


def test_job_manager_reports_nonzero_exit(tmp_path):
    import sys, time
    from mcp_agent import JobManager
    manager = JobManager(tmp_path)
    job = manager.start(name="failure", command=[sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path)
    for _ in range(100):
        status = manager.status(job.job_id)
        if status["status"] != "running":
            break
        time.sleep(0.02)
    assert status["status"] == "failed"
    assert status["returncode"] == 3


def test_prediction_worker_has_no_protocol_stdin(tmp_path, monkeypatch):
    import subprocess
    import mcp_agent
    calls = []
    monkeypatch.setattr(mcp_agent.subprocess, "Popen", lambda *a, **k: (calls.append(k) or SimpleNamespace(pid=1234)))
    mcp_agent.JobManager(tmp_path).start(name="test", command=["python"], cwd=tmp_path)
    assert calls[0]["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("worker_fails", [False, True])
def test_round_predicts_only_modified_and_preserves_all_rows(tmp_path, monkeypatch, worker_fails):
    import mcp_agent
    from pathlib import Path
    pool = tmp_path / "property"
    pool.mkdir()
    candidates = pd.DataFrame({"candidate_id": ["a", "b"], "SMILES": ["CCO", "CC"]})
    candidates.to_csv(tmp_path / "candidates.csv", index=False)
    affinity = candidates.assign(**{"Affinity [pKd]": [1.0, 2.0]})
    admet = candidates.assign(**{"[Example] Predictions": [1.0, 2.0]})
    affinity.to_csv(pool / "affinity.csv", index=False)
    admet.to_csv(pool / "admet.csv", index=False)
    def refine(source, target, **kwargs):
        frame = pd.read_csv(source)
        assert "Affinity [pKd]" in frame and "[Example] Predictions" in frame
        result = candidates.assign(Updated_SMILES=["CCN", "CC"], Property="Affinity [pKd]", Rationale="test", status=["modified", "unchanged"], error="")
        result.to_csv(target, index=False)
    monkeypatch.setattr(mcp_agent, "_refine_smiles", refine)
    def jobs(**kwargs):
        changed = pd.read_csv(kwargs["smiles_csv"])
        assert changed.candidate_id.tolist() == ["a"]
        changed.assign(**{"Affinity [pKd]": [3.0]}).to_csv(pool / "after-aff.csv", index=False)
        changed.assign(**{"[Example] Predictions": [3.0]}).to_csv(pool / "after-admet.csv", index=False)
        return [SimpleNamespace(job_id="a"), SimpleNamespace(job_id="b")]
    monkeypatch.setattr(mcp_agent, "_run_prediction_jobs", jobs)
    monkeypatch.setattr(mcp_agent, "_wait_for_jobs", lambda *a, **k: {"a": {"returncode": 1 if worker_fails else 0}})
    monkeypatch.setattr(mcp_agent, "_find_prediction_outputs", lambda *a: (pool / "after-aff.csv", pool / "after-admet.csv"))
    result = mcp_agent._optimize_iteration(candidates_csv=tmp_path / "candidates.csv", affinity_csv=pool / "affinity.csv", admet_csv=pool / "admet.csv", iter_dir=tmp_path, sequence="ACDE", protein="BCL-2", model="deepseek-flash", jm=None, progress=None, iteration=0)
    assert result["ok"] is not worker_fails
    output = pd.read_csv(result["comparison_csv"])
    assert output.candidate_id.tolist() == ["a", "b"]
    assert output.prediction_source.tolist() == (["unavailable", "reused"] if worker_fails else ["predicted", "reused"])


def test_baseline_missing_row_records_all_candidates(tmp_path):
    import mcp_agent
    candidates = pd.DataFrame({"candidate_id": ["a", "b"], "SMILES": ["CCO", "CC"]})
    candidates.to_csv(tmp_path / "candidates.csv", index=False)
    candidates.assign(**{"Affinity [pKd]": [1.0, 2.0]}).to_csv(tmp_path / "aff.csv", index=False)
    candidates.iloc[:1].assign(**{"[Example] Predictions": [1.0]}).to_csv(tmp_path / "admet.csv", index=False)
    result = mcp_agent._optimize_iteration(candidates_csv=tmp_path / "candidates.csv", affinity_csv=tmp_path / "aff.csv", admet_csv=tmp_path / "admet.csv", iter_dir=tmp_path, sequence="ACDE", protein="BCL2", model="deepseek-flash", jm=None, progress=None, iteration=0)
    assert not result["ok"]
    frame = pd.read_csv(result["comparison_csv"])
    assert frame.candidate_id.tolist() == ["a", "b"]
    assert frame.status.tolist() == ["failed", "failed"]


def test_interrupted_launcher_records_candidate_states(tmp_path):
    from scripts.check_pipeline import record_interrupted_candidates
    pool = tmp_path / "iter_00/pool"
    pool.mkdir(parents=True)
    pd.DataFrame({"candidate_id": ["a", "b"], "SMILES": ["CCO", "CC"]}).to_csv(pool / "candidates.csv", index=False)
    record_interrupted_candidates(tmp_path, "deadline")
    result = pd.read_csv(tmp_path / "iter_00/property/comparison.csv")
    assert result.candidate_id.tolist() == ["a", "b"]
    assert result.status.tolist() == ["failed", "failed"]
