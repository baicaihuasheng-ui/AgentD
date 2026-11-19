"""Candidate identity, validated prediction joins and measured comparisons."""
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Invalid SMILES")
    return Chem.MolToSmiles(molecule)


def identify(frame):
    frame = frame.copy()
    if "SMILES" not in frame or frame.empty:
        raise ValueError("Candidates require nonempty SMILES")
    identities = [canonical(smile) for smile in frame.SMILES]
    if "candidate_id" not in frame:
        frame.insert(0, "candidate_id", [f"c{i+1:04d}-" + hashlib.sha256(s.encode()).hexdigest()[:8] for i, s in enumerate(identities)])
    if frame.candidate_id.isna().any() or frame.candidate_id.duplicated().any() or (frame.candidate_id.astype(str).str.strip() == "").any():
        raise ValueError("Candidate IDs must be unique and nonempty")
    return frame


def aligned_prediction(candidates, prediction):
    prediction = prediction.copy()
    if "SMILES" not in prediction:
        raise ValueError("Prediction requires SMILES")
    if "candidate_id" not in prediction:
        # Compatibility for historical CSVs with unique molecular identities.
        lookup = candidates.assign(_canonical=candidates.SMILES.map(canonical))
        if lookup._canonical.duplicated().any():
            raise ValueError("Historical prediction lacks IDs for duplicate molecules")
        mapping = dict(zip(lookup._canonical, lookup.candidate_id))
        prediction["candidate_id"] = prediction.SMILES.map(canonical).map(mapping)
    prediction = identify(prediction)
    if set(prediction.candidate_id) != set(candidates.candidate_id):
        raise ValueError("Prediction candidate IDs do not match input")
    prediction = prediction.set_index("candidate_id").loc[candidates.candidate_id].reset_index()
    if [canonical(s) for s in prediction.SMILES] != [canonical(s) for s in candidates.SMILES]:
        raise ValueError("Prediction SMILES do not match candidate IDs")
    return prediction


def merge_predictions(candidates, affinity, admet):
    candidates = identify(candidates).reset_index(drop=True)
    affinity = aligned_prediction(candidates, affinity)
    admet = aligned_prediction(candidates, admet)
    values = pd.to_numeric(affinity["Affinity [pKd]"], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite affinity prediction")
    prediction_columns = [c for c in admet if c.startswith("Predictions_") or c.endswith("] Predictions")]
    if not prediction_columns or admet[prediction_columns].isna().any().any():
        raise ValueError("ADMET prediction values missing")
    numeric = admet[prediction_columns].select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite ADMET prediction")
    result = candidates.copy()
    for frame in (affinity, admet):
        for column in frame:
            if column in candidates:
                if column not in {"SMILES", "candidate_id"} and not frame[column].equals(candidates[column]):
                    raise ValueError(f"Input column changed in predictions: {column}")
                continue
            if column in result and not result[column].equals(frame[column]):
                raise ValueError(f"Conflicting prediction column: {column}")
            result[column] = frame[column].to_numpy()
    return result


def molecular_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return {"QED": QED.qed(molecule), "MW": Descriptors.MolWt(molecule), "LogP": Descriptors.MolLogP(molecule)}


def comparison_table(before, refinement, after):
    before = identify(before)
    refinement = aligned_prediction(before, refinement)
    after = after.set_index("candidate_id") if not after.empty else pd.DataFrame()
    if not after.empty and after.index.duplicated().any():
        raise ValueError("Duplicate after-prediction IDs")
    rows = []
    columns = [c for c in before if c == "Affinity [pKd]" or c.startswith("Predictions_") or c.endswith("] Predictions") or c.endswith("] Probability")]
    for (_, original), (_, change) in zip(before.iterrows(), refinement.iterrows()):
        row = change.to_dict()
        candidate_id = original.candidate_id
        updated = None
        if change.status == "unchanged":
            updated = original
            row["prediction_source"] = "reused"
        elif change.status == "modified" and candidate_id in after.index:
            updated = after.loc[candidate_id]
            if canonical(updated.SMILES) != canonical(change.Updated_SMILES):
                raise ValueError("Updated prediction molecule mismatch")
            row["prediction_source"] = "predicted"
        else:
            row["prediction_source"] = "unavailable"
            if change.status == "modified":
                row["status"] = "prediction_failed"
                row["error"] = "Updated prediction unavailable"
        old_values = {**original.to_dict(), **molecular_properties(original.SMILES)}
        new_values = {**updated.to_dict(), **molecular_properties(updated.SMILES)} if updated is not None else {}
        for column in columns + ["QED", "MW", "LogP"]:
            old, new = old_values.get(column), new_values.get(column)
            row["before_" + column] = old
            row["after_" + column] = new
            if isinstance(old, (int, float, np.number)) and isinstance(new, (int, float, np.number)):
                row["delta_" + column] = new - old
        rows.append(row)
    return pd.DataFrame(rows)
