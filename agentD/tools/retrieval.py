import logging
import re
# Built-in and standard library
import os
import time
import json
import requests
from collections import defaultdict

# Third-party libraries (LangChain, ChEMBL)
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from agentD.providers.deepseek import get_chat_model
from langchain_community.document_loaders import PyPDFLoader
from agentD.local_embeddings import get_embeddings, split_documents
from agentD.providers.deepseek_search import DeepSeekSearch
from agentD.documents import collect_documents,load_documents
from langchain_community.vectorstores import FAISS
# Local utilities
from agentD.utils import download_pdf, RetrievalQABypassTokenLimit
from configs.tool_globals import UNIPROT_NUM_IDS, MAX_PAPERS, EMBEDDING_MODEL, PAPER_DIR
# breakpoint()
# UNIPROT_NUM_IDS = 1
# MAX_PAPERS = 10 # Number of papers to download for molecule optimization guideline
# PAPER_DIR = "papers"  # Directory to store downloaded papers
# EMBEDDING_MODEL = 'text-embedding-3-large' 

@tool
def search(query):
    """Search current webpages using DeepSeek; query must contain 1–128 characters. Returns titles, URLs and labeled source text."""
    hits=DeepSeekSearch().search(query)
    from agentD.call_budget import current_budget
    if current_budget():
        hits = [{"title": hit["title"], "url": hit["url"],
                 "text": (hit["content"] or hit["summary"] or hit["snippet"])[:1500],
                 "excerpt_limit_characters": 1500} for hit in hits]
    return json.dumps(hits,ensure_ascii=False)


@tool
def get_uniprot_ids(protein_name: str):
# def get_uniprot_info(protein_name: str):
    """
    Searches for UniProt IDs based on a given protein name and returns the IDs.
    Strictly pass the protein name itself, like "TP53" or "EGFR".
    """
    import requests

    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"{protein_name} AND organism_id:9606"  # Ensure the query is formatted properly
    # size  = 5
    params = {
        "query": query,
        "format": "json",
        "size": UNIPROT_NUM_IDS  # Number of results to retrieve
    }

    #print(f"Sending API request: {base_url}?query={query}&format=json&size={size}")  # Debugging output

    try:
        response = requests.get(base_url, params=params, timeout=(10, 60))
        response.raise_for_status()  # Raise an error for HTTP failures
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            #return data['results'][0]['primaryAccession'] #[entry["primaryAccession"] for entry in data["results"]][0]  # Return the first UniProt ID
            matches = []
            for entry in data["results"]:
                accession = entry.get("primaryAccession")
                if not accession:
                    continue
                description = entry.get("proteinDescription") or {}
                names = [description.get("recommendedName") or {}]
                names.extend(description.get("submissionNames") or [])
                name = next((item.get("fullName", {}).get("value")
                             for item in names
                             if item.get("fullName", {}).get("value")), accession)
                matches.append((accession, name))
            return matches
        else:
            return f"No UniProt IDs found for '{protein_name}'."

    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    
@tool
def fetch_uniprot_fasta(uniprot_id):
    """
    Fetch the FASTA sequence using the UniProt ID.
    """
    import re

    if not isinstance(uniprot_id, str):
        raise ValueError("UniProt accession must be a string")
    uniprot_id = uniprot_id.strip()
    uniprot_id = uniprot_id.strip('"')
    uniprot_id = uniprot_id.strip()
    accession_pattern = (
        r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
        r"(?:-[1-9][0-9]*)?"
    )
    if not re.fullmatch(accession_pattern, uniprot_id):
        raise ValueError("Invalid UniProt accession")
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(fasta_url, timeout=(10, 60))
    
    if response.status_code == 200:
        return response.text
    else:
        logging.getLogger(__name__).warning(f"Failed to fetch FASTA for {uniprot_id}. Status Code: {response.status_code}")
        return None


@tool
def save_results(input_str: str) -> str:
    """
    Save content to a file. Input should be a valid JSON string.
    """
    
    try:
        input_str = input_str.strip()
        data = json.loads(input_str)
        filename = "extraction.json" #data["save_name"]+".json"
  
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4))
        return f"Saved to {filename}"

    except json.JSONDecodeError:
        # Fallback: save raw input as text
        fallback_filename = "extraction.txt"
        with open(fallback_filename, "w", encoding="utf-8") as f:
            f.write(input_str)
        return f"Saved raw input to {fallback_filename}"
    
    except Exception as e:
        return f"Failed to write file: {e}"


def _load_chembl_client():
    # One retry loop owns failures, including the import-time /spore request.
    from chembl_webresource_client.settings import Settings
    settings = Settings.Instance()
    settings.TOTAL_RETRIES = 0
    settings.NEW_CLIENT_TIMEOUT = (10, 30)
    from chembl_webresource_client.new_client import new_client
    return new_client


@tool
def get_drug_smiles(drug_name: str):
    """
    Fetch canonical SMILES for one drug name from ChEMBL, or None if not found.
    """
    # ReAct action inputs may retain trailing newlines or enclosing quotes.
    drug_name = drug_name.strip()
    if len(drug_name) >= 2 and drug_name[0] == drug_name[-1] and drug_name[0] in ('"', "'"):
        drug_name = drug_name[1:-1].strip()
    if not drug_name or '\n' in drug_name or '\r' in drug_name:
        raise ValueError('drug_name must contain one non-empty drug name')
    logger = logging.getLogger(__name__)
    max_attempts = 6  # Initial attempt followed by at most five retries.
    for attempt in range(1, max_attempts + 1):
        logger.warning('ChEMBL client lookup attempt=%s/6 event=started', attempt)
        try:
            client = _load_chembl_client()
            mols = client.molecule.filter(pref_name__iexact=drug_name)
            # The SDK fetches records lazily, so iteration must be retried too.
            smiles = None
            for mol in mols:
                structures = mol.get('molecule_structures') or {}
                candidate = structures.get('canonical_smiles')
                if isinstance(candidate, str) and candidate.strip():
                    smiles = candidate
                    break
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status is None:
                match = re.search(r'\bstatus\s+(\d{3})\b', str(error))
                status = match.group(1) if match else None
            logger.warning('ChEMBL client lookup attempt=%s/6 event=failed status=%s error_type=%s',
                           attempt, status, type(error).__name__)
            if attempt == max_attempts:
                detail = f'HTTP {status}' if status is not None else type(error).__name__
                raise RuntimeError(
                    f'ChEMBL client lookup failed after 6 attempts '
                    f'(initial attempt + 5 retries; last error: {detail})'
                ) from None
            logger.warning('ChEMBL client lookup retry=%s/5 delay_s=2', attempt)
            time.sleep(2)
        else:
            logger.warning('ChEMBL client lookup attempt=%s/6 event=completed', attempt)
            if smiles is None:
                logger.warning('No SMILES found for %s.', drug_name)
            return smiles




@tool
def question_answering(query: str):
    """Answer a question using locally downloaded PDFs/Markdown and local embeddings; cite available source URLs."""
    if not os.path.isdir(PAPER_DIR):
        return "No papers found. Collect source documents first."
    documents=load_documents(PAPER_DIR)
    if not documents:
        return "No readable source documents found."
    vectorstore=FAISS.from_documents(split_documents(documents),get_embeddings())
    result=RetrievalQABypassTokenLimit(vectorstore,query,get_chat_model(temperature=0.3),return_sources=True)
    sources=sorted({str(meta.get('source','')) for meta in result['sources']})
    return json.dumps({'answer':result['answer'],'available_sources':sources,'retrieved_sources':result['sources']},ensure_ascii=False)


@tool
def download_relevant_papers(query: str):
    """Find relevant literature through DeepSeek Search and save accessible PDF/web text with source links. Query must be <=128 characters."""
    return collect_documents(query,PAPER_DIR,DeepSeekSearch(),max_papers=MAX_PAPERS)
