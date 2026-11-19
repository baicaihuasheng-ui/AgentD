import json
from pathlib import Path

import pandas as pd
import pytest
import requests


@pytest.fixture(autouse=True)
def no_real_polling_wait(monkeypatch):
    monkeypatch.setattr("agentD.deeppk.time.sleep", lambda seconds: None)


@pytest.mark.parametrize("error_type", [requests.ReadTimeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError])
def test_transport_failure_retries_same_query(tmp_path, monkeypatch, error_type):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient(interval=0)
    from unittest.mock import Mock
    post = Mock(return_value=response({"job_id": "known-job"}))
    monkeypatch.setattr(client.session, "post", post)
    send = Mock(side_effect=[error_type("test"), response({"0": {"SMILES": "CCO", "Predictions_logp": 1.2}})])
    monkeypatch.setattr(client.session, "send", send)
    sleeps = []
    monkeypatch.setattr("agentD.deeppk.time.sleep", sleeps.append)
    assert client.predict(source, tmp_path / "out").is_file()
    assert post.call_count == 1 and send.call_count == 2 and sleeps == [2]
    assert all(call.args[0].method == "GET" and b"known-job" in call.args[0].body for call in send.call_args_list)


def test_transport_failures_stop_on_fifth_and_save_job(tmp_path, monkeypatch):
    from agentD.deeppk import DeepPKClient
    from unittest.mock import Mock
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient(interval=0)
    post = Mock(return_value=response({"job_id": "known-job"}))
    send = Mock(side_effect=requests.exceptions.ChunkedEncodingError("partial body"))
    monkeypatch.setattr(client.session, "post", post)
    monkeypatch.setattr(client.session, "send", send)
    sleeps = []
    monkeypatch.setattr("agentD.deeppk.time.sleep", sleeps.append)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        client.predict(source, tmp_path / "out")
    assert post.call_count == 1 and send.call_count == 5
    assert sleeps == [2, 4, 8, 16]
    state = json.loads(next((tmp_path / "out").glob("*.state.json")).read_text())
    assert state["query_failures"] == 5 and state["status"] == "failed"
    assert state["job_id"] == "known-job"
    assert not (tmp_path / "out/admet_input.csv").exists()
    monkeypatch.setattr(client.session, "send", lambda *a, **k: response({"0": {"SMILES": "CCO", "Predictions_logp": 1.2}}))
    assert client.predict(source, tmp_path / "out").is_file()
    assert post.call_count == 1


def test_backoff_respects_remaining_deadline(tmp_path, monkeypatch):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient(interval=0, max_wait=1)
    monkeypatch.setattr(client.session, "post", lambda *a, **k: response({"job_id": "known-job"}))
    clock = [0]
    monkeypatch.setattr("agentD.deeppk.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("agentD.deeppk.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    calls = []
    def send(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ChunkedEncodingError("partial")
    monkeypatch.setattr(client.session, "send", send)
    with pytest.raises(TimeoutError):
        client.predict(source, tmp_path / "out")
    assert calls == [1] and clock[0] == 1


def response(payload):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(payload).encode()
    return result


def test_deeppk_default_wait_is_fifteen_minutes():
    from agentD.deeppk import DeepPKClient
    assert DeepPKClient().max_wait == 900


def test_prediction_tool_passes_fifteen_minute_wait(monkeypatch):
    from unittest.mock import Mock
    from agentD import deeppk
    from agentD.tools import prediction
    factory = Mock()
    factory.return_value.predict.return_value = Path("admet_example.csv")
    monkeypatch.setattr(deeppk, "DeepPKClient", factory)
    prediction.get_admet_predictions.invoke({"csv_file_path": "example.csv"})
    assert factory.call_args.kwargs["max_wait"] == 900


@pytest.mark.parametrize("encoded", [False, True])
def test_deeppk_maps_reordered_rows_and_preserves_confidence(tmp_path, monkeypatch, encoded):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"ID": ["first", "second", "third"], "SMILES": ["CCO", "CC", "CCO"]}).to_csv(source, index=False)
    payload = {"1": {"SMILES": "CC", "Predictions_logp": 2, "Probability_logp": 0.9},
               "0": {"SMILES": "CCO", "Predictions_logp": 1, "Probability_logp": 0.8}}
    replies = iter([response({"status": "running"}), response(json.dumps(payload) if encoded else payload)])
    submitted = []
    client = DeepPKClient(interval=0)
    monkeypatch.setattr(client.session, "post", lambda *a, **k: (submitted.append(k) or response({"job_id": "test-1"})))
    monkeypatch.setattr(client.session, "send", lambda *a, **k: next(replies))
    output = client.predict(source, tmp_path / "out")
    frame = pd.read_csv(output)
    assert frame.ID.tolist() == ["first", "second", "third"]
    assert frame.Predictions_logp.tolist() == [1, 2, 1]
    assert frame.Probability_logp.tolist() == [0.8, 0.9, 0.8]
    assert submitted[0]["timeout"] == (10, 60)
    # Completed jobs are reused without either submission or polling.
    assert client.predict(source, tmp_path / "out") == output
    assert len(submitted) == 1


def test_deeppk_timeout_resumes_same_job(tmp_path, monkeypatch):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient(interval=0)
    submissions = []
    monkeypatch.setattr(client.session, "post", lambda *a, **k: (submissions.append(1) or response({"job_id": "known-job"})))
    def timeout(*a, **k):
        raise requests.ReadTimeout("test")
    polls = []
    def recorded_timeout(*args, **kwargs):
        polls.append(kwargs)
        return timeout(*args, **kwargs)
    monkeypatch.setattr(client.session, "send", recorded_timeout)
    with pytest.raises(requests.ReadTimeout):
        client.predict(source, tmp_path / "out")

    state = json.loads(next((tmp_path / "out").glob("*.state.json")).read_text())
    assert state["job_id"] == "known-job" and state["status"] == "failed"
    assert len(polls) == 5
    assert all(call["timeout"][1] == 60 for call in polls)
    monkeypatch.setattr(client.session, "send", lambda *a, **k: response({"0": {"SMILES": "CCO", "Predictions_logp": 1.1}}))
    client.predict(source, tmp_path / "out")
    assert len(submissions) == 1


def test_deeppk_recovers_after_read_timeout_without_resubmission(tmp_path, monkeypatch):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient(interval=0)
    submissions = []
    monkeypatch.setattr(client.session, "post", lambda *a, **k: (submissions.append(1) or response({"job_id": "same-job"})))
    replies = iter([requests.ReadTimeout("test"), response({"0": {"SMILES": "CCO", "Predictions_logp": 1.1}})])
    queried = []
    def send(request, **kwargs):
        queried.append(request.body)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(client.session, "send", send)
    output = client.predict(source, tmp_path / "out")
    assert pd.read_csv(output).Predictions_logp.tolist() == [1.1]
    assert len(submissions) == 1 and len(queried) == 2
    assert all(b"same-job" in body for body in queried)


def test_deeppk_retry_preserves_total_deadline(tmp_path, monkeypatch):
    import agentD.deeppk as module
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = module.DeepPKClient(interval=0, max_wait=100)
    monkeypatch.setattr(client.session, "post", lambda *a, **k: response({"job_id": "same-job"}))
    clock = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    polls = []
    def timeout(request, **kwargs):
        wait = kwargs["timeout"][1]
        polls.append(wait)
        clock[0] += wait
        raise requests.ReadTimeout("test")
    monkeypatch.setattr(client.session, "send", timeout)
    with pytest.raises(TimeoutError, match="total polling"):
        client.predict(source, tmp_path / "out")
    assert polls == [60, 40]


@pytest.mark.parametrize("error_type", [requests.ReadTimeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError])
def test_ambiguous_submission_is_not_repeated(tmp_path, monkeypatch, error_type):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient()
    def timeout(*a, **k):
        raise error_type("test")
    monkeypatch.setattr(client.session, "post", timeout)
    with pytest.raises(error_type):
        client.predict(source, tmp_path / "out")
    with pytest.raises(RuntimeError, match="unknown"):
        client.predict(source, tmp_path / "out")


def test_previous_process_lock_file_does_not_block_job_recovery(tmp_path, monkeypatch):
    import hashlib
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = tmp_path / "out"
    directory.mkdir()
    state_path = directory / (source.stem + "-" + fingerprint[:16] + ".state.json")
    state_path.write_text(json.dumps({"input_sha256": fingerprint, "job_id": "known-job", "status": "running"}))
    state_path.with_suffix(".lock").write_text("999999")
    client = DeepPKClient()
    monkeypatch.setattr(client.session, "send", lambda *a, **k: response({"0": {"SMILES": "CCO", "Predictions_logp": 1}}))
    assert client.predict(source, directory).is_file()


@pytest.mark.parametrize("result,expected", [("ADMET predictions saved to example.csv", "passed"), ({"error": "test"}, "failed")])
def test_deeppk_acceptance_status_is_final(tmp_path, monkeypatch, result, expected):
    import sys
    from types import SimpleNamespace
    from agentD.tools import prediction
    from scripts.verify_affinity import main
    monkeypatch.setattr(prediction, "get_admet_predictions", SimpleNamespace(invoke=lambda *a, **k: result))
    monkeypatch.setattr(sys, "argv", ["verify_affinity.py", "--deeppk", "--output", str(tmp_path)])
    assert main() == (0 if expected == "passed" else 1)
    assert json.loads((tmp_path / "acceptance.json").read_text())["status"] == expected


def test_current_deeppk_labels_preserve_prediction_and_probability():
    from agentD.deeppk import align_results
    label = "[Absorption/Human Intestinal Absorption]"
    inputs = pd.DataFrame({"ID": ["ethanol"], "SMILES": ["CCO"]})
    payload = {"0": {"SMILES": "CCO", label + " Predictions": "Absorbed", label + " Probability": 0.9}}
    output = align_results(payload, inputs)
    assert output[label + " Predictions"].tolist() == ["Absorbed"]
    assert output[label + " Probability"].tolist() == [0.9]


@pytest.mark.parametrize("payload", [{"error": "failed"}, {}, {"0": {"SMILES": "CC", "Predictions_logp": 1}}])
def test_bad_or_unmapped_results_rejected(tmp_path, monkeypatch, payload):
    from agentD.deeppk import DeepPKClient
    source = tmp_path / "input.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    client = DeepPKClient()
    monkeypatch.setattr(client.session, "post", lambda *a, **k: response({"job_id": "test"}))
    monkeypatch.setattr(client.session, "send", lambda *a, **k: response(payload))
    with pytest.raises(ValueError):
        client.predict(source, tmp_path / "out")
    assert not (tmp_path / "out/admet_input.csv").exists()
