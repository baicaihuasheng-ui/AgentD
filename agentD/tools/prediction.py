from langchain.tools import tool
from rdkit import Chem
import json, os, random
import pandas as pd
from configs.tool_globals import ADMET_WAIT_INTERVAL, ADMET_MAX_WAIT_TIME, PROPERTY_DIR

random.seed(2102)

# Adapted with modifications from the original implementation at:
# https://github.com/mehradans92/dZiner/blob/main/dziner/tools.py
@tool
def check_smiles_validity(smiles: str):
    '''
    Checks if the provided SMILES string is valid.
    '''
    try:
        smiles = smiles.strip()
        smiles = smiles.replace("\n", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES" 
        return "Valid SMILES" 
    except Exception as e:
        return f"Error checking SMILES validity: {str(e)}"


@tool
def predict_affinity_batch(data: str):
    """Predict affinities from JSON containing sequence and smiles_path (CSV SMILES column)."""
    from pathlib import Path
    from agentD.affinity import AffinityPredictor, atomic_json
    try:
        payload = json.loads(data)
        frame = pd.read_csv(payload["smiles_path"])
        predictor = AffinityPredictor()
        frame["Affinity [pKd]"] = predictor.predict(payload["sequence"], frame["SMILES"].tolist())
        output = Path(PROPERTY_DIR) / ("affinity_" + Path(payload["smiles_path"]).name)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
        atomic_json(output.with_suffix(".metrics.json"), predictor.report())
        return "Affinity predictions saved to ", str(output)
    except Exception as error:
        return f"Error predicting affinity: {type(error).__name__}: {error}"


@tool
def get_admet_predictions(csv_file_path: str):
    """Submit or resume a DeepPK ADMET job for a CSV with a SMILES column."""
    from agentD.deeppk import DeepPKClient
    try:
        output = DeepPKClient(interval=ADMET_WAIT_INTERVAL, max_wait=ADMET_MAX_WAIT_TIME).predict(csv_file_path, PROPERTY_DIR)
        return f"ADMET predictions saved to {output}"
    except Exception as error:
        return {"error": f"DeepPK failed: {type(error).__name__}: {error}"}
