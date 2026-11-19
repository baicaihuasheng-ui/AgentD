"""Durable DeepPK jobs with bounded polling and explicit molecule alignment."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import pandas as pd
import requests
from rdkit import Chem
from filelock import FileLock

from agentD.affinity import atomic_json

URL = "https://biosig.lab.uq.edu.au/deeppk/api/predict"


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("DeepPK input/result contains invalid SMILES")
    return Chem.MolToSmiles(molecule)


def align_results(payload, inputs):
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        if not payload or not all(isinstance(row, dict) for row in payload.values()):
            raise ValueError("Unexpected DeepPK response")
        rows = list(payload.values())
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Unexpected DeepPK response")
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or "SMILES" not in row:
            raise ValueError("DeepPK result lacks molecular identity")
        if not any((key.startswith("Predictions_") or (key.startswith("[") and key.endswith("] Predictions")))
                   and value is not None for key, value in row.items()):
            raise ValueError("DeepPK result lacks predictions")
        identity = canonical(row["SMILES"])
        if identity in mapping and mapping[identity] != row:
            raise ValueError("Conflicting duplicate DeepPK predictions")
        mapping[identity] = row
    requested = [canonical(smiles) for smiles in inputs.SMILES]
    if set(mapping) != set(requested):
        raise ValueError("DeepPK results do not match submitted molecules")
    aligned = pd.DataFrame([mapping[key] for key in requested]).drop(columns="SMILES")
    if set(aligned.columns) & set(inputs.columns):
        raise ValueError("DeepPK output columns conflict with input columns")
    return pd.concat([inputs.reset_index(drop=True), aligned], axis=1)


class DeepPKClient:
    def __init__(self, interval=30, max_wait=900):
        if interval < 0 or max_wait <= 0:
            raise ValueError("Invalid DeepPK polling limits")
        self.interval = interval
        self.max_wait = max_wait
        self.session = requests.Session()

    def predict(self, source, directory):
        source, directory = Path(source), Path(directory)
        content = source.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        inputs = pd.read_csv(io.BytesIO(content))
        if "SMILES" not in inputs or inputs.empty:
            raise ValueError("DeepPK requires a nonempty SMILES column")
        for smile in inputs.SMILES:
            canonical(smile)
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / (source.stem + "-" + fingerprint[:16] + ".state.json")
        lock = state_path.with_suffix(".lock")
        # OS-backed locks are released even when the process crashes.
        with FileLock(str(lock), timeout=0):
            return self._run(inputs, fingerprint, source, directory, state_path)

    def _run(self, inputs, fingerprint, source, directory, state_path):
        state = json.loads(state_path.read_text()) if state_path.exists() else {"input_sha256": fingerprint}
        output = directory / ("admet_" + source.name)
        if state.get("status") == "completed" and output.is_file():
            if hashlib.sha256(output.read_bytes()).hexdigest() == state.get("output_sha256"):
                return output
        try:
            if not state.get("job_id"):
                if state.get("status"):
                    raise RuntimeError("Submission outcome unknown. Check saved state before submitting again")
                state["status"] = "submitting"
                atomic_json(state_path, state)
                # Only SMILES are sent. Local input IDs are retained for output alignment.
                data = inputs[["SMILES"]].to_csv(index=False).encode()
                reply = self.session.post(URL, files={"smiles_file": ("molecules.csv", data, "text/csv")},
                                          data={"pred_type": "admet"}, timeout=(10, 60), allow_redirects=False)
                reply.raise_for_status()
                submitted = reply.json()
                job_id = submitted.get("job_id") if isinstance(submitted, dict) else None
                if not isinstance(job_id, str) or not job_id.strip():
                    raise ValueError("DeepPK did not return a job_id")
                state.update(job_id=job_id, status="submitted")
                atomic_json(state_path, state)
            deadline = time.monotonic() + self.max_wait
            read_timeouts = 0
            query_failures = 0
            state.update(read_timeouts=0, query_failures=0)
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                request = requests.Request("GET", URL, files={"job_id": (None, state["job_id"])})
                try:
                    reply = self.session.send(self.session.prepare_request(request), timeout=(min(10, remaining), min(60, remaining)), allow_redirects=False)
                except (requests.ReadTimeout, requests.ConnectionError,
                        requests.exceptions.ChunkedEncodingError) as error:
                    query_failures += 1
                    read_timeouts += int(isinstance(error, requests.ReadTimeout))
                    state.update(status="retrying", read_timeouts=read_timeouts,
                                 query_failures=query_failures,
                                 last_query_error=type(error).__name__)
                    atomic_json(state_path, state)
                    if query_failures >= 5:
                        raise
                    time.sleep(min(2 ** query_failures,
                                   max(0, deadline - time.monotonic())))
                    continue
                reply.raise_for_status()
                payload = reply.json()
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict) and payload.get("status") == "running":
                    state["status"] = "running"
                    atomic_json(state_path, state)
                    time.sleep(min(self.interval, max(0, deadline - time.monotonic())))
                    continue
                atomic_json(state_path.with_suffix(".result.json"), payload)
                aligned = align_results(payload, inputs)
                temporary = output.with_suffix(".csv.tmp")
                aligned.to_csv(temporary, index=False)
                os.replace(temporary, output)
                state.update(status="completed", output=str(output),
                             output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                state.pop("error", None)
                atomic_json(state_path, state)
                return output
            raise TimeoutError("DeepPK total polling time exceeded")
        except Exception as error:
            state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            atomic_json(state_path, state)
            raise
