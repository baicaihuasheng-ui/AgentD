"""Offline E5 embeddings shared by agent memory and document retrieval."""

from functools import lru_cache
from pathlib import Path
from langchain_core.embeddings import Embeddings
from agentD.runtime_config import load_runtime


class LocalE5Embeddings(Embeddings):
    def __init__(self, path, device="cpu", batch_size=8):
        self.path = Path(path).resolve()
        self.device = device
        self.batch_size = batch_size

    def _load(self):
        if not self.path.is_dir():
            raise FileNotFoundError(f"Offline embedding model missing: {self.path}")
        return _model(str(self.path), self.device)

    @property
    def tokenizer(self):
        return self._load()[0]

    @property
    def dimension(self):
        return self._load()[1].config.hidden_size

    def _encode(self, texts, prefix):
        if not texts:
            return []
        import torch

        tokenizer, model = self._load()
        result = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [
                    prefix + text for text in texts[start : start + self.batch_size]
                ]
                encoded = tokenizer(
                    batch,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].bool()
                hidden = hidden.masked_fill(~mask[..., None], 0.0)
                pooled = hidden.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)
                result.extend(
                    torch.nn.functional.normalize(pooled, p=2, dim=1)
                    .cpu()
                    .float()
                    .tolist()
                )
        return result

    def embed_query(self, text):
        return self._encode([text], "query: ")[0]

    def embed_documents(self, texts):
        return self._encode(texts, "passage: ")


@lru_cache(maxsize=2)
def _model(path, device):
    import torch
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        .to(device)
        .eval()
    )
    return tokenizer, model


@lru_cache(maxsize=2)
def _adapter(path, device):
    return LocalE5Embeddings(path, device)


def get_embeddings():
    cfg = load_runtime()
    return _adapter(str(cfg.embedding_path), cfg.embedding_device)


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        get_embeddings().tokenizer, chunk_size=480, chunk_overlap=48
    )
    return splitter.split_documents(documents)
