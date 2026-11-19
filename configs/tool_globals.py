# global variables for retrieval task
UNIPROT_NUM_IDS = 1 # Number of UniProt IDs to retrieve
MAX_PAPERS = 10 # Number of papers to download for molecule optimization guideline
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
PAPER_DIR = "papers"

# global variables for generation task
POOL_PATH = "pool"
from agentD.runtime_config import load_runtime
REINVENT_PATH = str(load_runtime().reinvent_path)

# global variables for prediction task
ADMET_WAIT_INTERVAL = 30
ADMET_MAX_WAIT_TIME = 900
PROPERTY_DIR = "property"
