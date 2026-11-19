"""Sequential, local-only BAPULM inference retaining upstream feature semantics."""
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import psutil
import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from agentD.runtime_config import load_runtime
from agentD.tools.BAPULM import BAPULM, set_seed

PREPROCESS_VERSION = "bapulm-mean-special-tokens-v1"


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def measure_stage(metrics, name, device):
    """RSS sampled at 50 ms, CUDA allocator peak measured for the selected device."""
    started = time.perf_counter()
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    worker = threading.Thread(target=sample, daemon=True)
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        peak[0] = max(peak[0], process.memory_info().rss)
        metrics.append({"stage": name, "device": device, "elapsed_seconds": time.perf_counter() - started,
                        "peak_process_rss_bytes": peak[0],
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0})


class AffinityPredictor:
    def __init__(self, config=None):
        self.config = config or load_runtime()
        self.device = self.config.affinity_device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.config.affinity_precision not in {"float16", "float32"}:
            raise ValueError("affinity_precision must be float16 or float32")
        self.metrics = []
        self.cache_hit = False

    def _protein(self, sequence, device):
        model = tokenizer = tokens = None
        dtype = torch.float16 if device.startswith("cuda") and self.config.affinity_precision == "float16" else torch.float32
        try:
            with measure_stage(self.metrics, "protein", device):
                tokenizer = T5Tokenizer.from_pretrained(str(self.config.protein_model_path), do_lower_case=False, local_files_only=True)
                model = T5EncoderModel.from_pretrained(str(self.config.protein_model_path), torch_dtype=dtype,
                                                      low_cpu_mem_usage=True, local_files_only=True).to(device).eval()
                tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=3200).to(device)
                with torch.inference_mode():
                    return model(**tokens).last_hidden_state.mean(dim=1).float().cpu()
        finally:
            del model, tokenizer, tokens
            release_memory()

    def _molecules(self, smiles):
        # Cache hits must not change the random-feature initialization of MoLFormer.
        set_seed(2102)
        model = tokenizer = tokens = None
        try:
            with measure_stage(self.metrics, "molecules", self.device):
                # This is audited compat-v4 code from the fixed local snapshot.
                tokenizer = AutoTokenizer.from_pretrained(str(self.config.molecule_model_path), deterministic_eval=True,
                                                          trust_remote_code=True, local_files_only=True)
                model = AutoModel.from_pretrained(str(self.config.molecule_model_path), trust_remote_code=True,
                                                  local_files_only=True).to(self.device).eval()
                features = []
                with torch.inference_mode():
                    for smile in smiles:
                        tokens = tokenizer(smile, return_tensors="pt", padding=True, truncation=True, max_length=278).to(self.device)
                        features.append(model(**tokens).last_hidden_state.mean(dim=1).float().cpu())
                return torch.cat(features)
        finally:
            del model, tokenizer, tokens
            release_memory()

    def _protein_features(self, sequence, device):
        manifest = self.config.protein_model_path / "manifest.json"
        identity = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else str(self.config.protein_model_path)
        metadata = {"sequence": sequence, "model": identity, "preprocessing": PREPROCESS_VERSION,
                    "precision": self.config.affinity_precision if device.startswith("cuda") else "float32", "device": device, "seed": 2102}
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        cache = self.config.affinity_cache_path / (key + ".pt")
        if cache.is_file():
            feature = torch.load(cache, map_location="cpu", weights_only=True)
            if feature.shape != (1, 1024) or not torch.isfinite(feature).all():
                raise ValueError("Invalid protein feature cache")
            self.cache_hit = True
            self.protein_metadata = metadata
            return feature.float()
        feature = self._protein(sequence, device).float().cpu()
        if feature.shape != (1, 1024) or not torch.isfinite(feature).all():
            raise ValueError("Invalid protein embedding")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(key + "." + uuid.uuid4().hex + ".tmp")
        torch.save(feature, temporary)
        os.replace(temporary, cache)
        atomic_json(cache.with_suffix(".json"), metadata)
        self.protein_metadata = metadata
        return feature

    def predict(self, sequence, smiles):
        if not isinstance(sequence, str) or not re.fullmatch(r"[A-Za-z\s]+", sequence):
            raise ValueError("Protein sequence must contain amino-acid letters")
        self.protein_sequence = " ".join(re.sub(r"[UZOB]", "X", "".join(sequence.upper().split())))
        if not smiles or any(not isinstance(s, str) or not s.strip() or Chem.MolFromSmiles(s) is None for s in smiles):
            raise ValueError("Input contains invalid or empty SMILES")
        set_seed(2102)
        fallback = False
        try:
            self.protein_features = self._protein_features(self.protein_sequence, self.device)
        except torch.cuda.OutOfMemoryError:
            fallback = True
        if fallback:
            release_memory()
            self.protein_features = self._protein_features(self.protein_sequence, "cpu")
        molecules = self._molecules(smiles).float().cpu()
        if molecules.shape != (len(smiles), 768) or not torch.isfinite(molecules).all():
            raise ValueError("Invalid molecule embeddings")
        with measure_stage(self.metrics, "head", "cpu"):
            model = BAPULM().float().cpu().eval()
            checkpoint = torch.load(self.config.affinity_checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint, strict=True)
            with torch.inference_mode():
                values = model(self.protein_features.expand(len(smiles), -1), molecules).flatten()
                values = values * 1.5614094578916633 + 6.51286529169358
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite affinity output")
        return values.tolist()

    def report(self):
        return {"seed": 2102, "cache_hit": self.cache_hit, "protein": self.protein_metadata,
                "molecule_precision": "float32", "stages": self.metrics}
