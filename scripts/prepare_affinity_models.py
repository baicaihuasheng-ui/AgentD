"""Download immutable official snapshots and verify every LFS object."""
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODELS = [
    ("Rostlab/prot_t5_xl_half_uniref50-enc", "94a6abc029ae13029317b140b7424e012bf8dfbf",
     ["README.md", "config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]),
    ("ibm-research/MoLFormer-XL-both-10pct", "7b12d946c181a37f6012b9dc3b002275de070314",
     ["README.md", "config.json", "model.safetensors", "configuration_molformer.py", "modeling_molformer.py",
      "tokenization_molformer.py", "tokenization_molformer_fast.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json"]),
]
ROOT = Path("D:/CodexData/Downloads/agentd-models")
TEMP = Path("D:/CodexData/Temp/affinity-downloads")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ranged_download(url, partial, size):
    block = 4 * 1024 * 1024
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("Partial download exceeds expected size")
    def fetch(start):
        end = min(start + block, size) - 1
        for attempt in range(4):
            try:
                with requests.get(url + f"?range_start={start}", headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 45)) as reply:
                    reply.raise_for_status()
                    if reply.status_code != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server did not honor requested byte range")
                    body = reply.content
                    if len(body) != end - start + 1:
                        raise ValueError("Truncated byte range")
                    return body
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2)
    with partial.open("ab") as stream, ThreadPoolExecutor(max_workers=4) as pool:
        # Keep only four ranges in flight to bound memory.
        for start in range(offset, size, block * 4):
            positions = range(start, min(start + block * 4, size), block)
            for body in pool.map(fetch, positions):
                stream.write(body)
                stream.flush()
            print(partial.name, stream.tell(), "/", size, flush=True)


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    for repo, revision, files in MODELS:
        response = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=(15, 60))
        response.raise_for_status()
        metadata = response.json()
        assert metadata["sha"] == revision
        entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        target = ROOT / repo.split("/")[-1] / revision
        target.mkdir(parents=True, exist_ok=True)
        manifest = {"repository": repo, "revision": revision, "files": []}
        for name in files:
            entry = entries[name]
            expected = entry.get("lfs", {}).get("sha256")
            size = entry["size"]
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            output = target / name
            if not (output.exists() and output.stat().st_size == size and (not expected or digest(output) == expected)):
                partial = TEMP / (repo.split("/")[-1] + "-" + name + ".part")
                for attempt in range(5):
                    try:
                        if size > 8 * 1024 * 1024:
                            ranged_download(url, partial, size)
                            if digest(partial) != expected:
                                partial.unlink()
                                raise ValueError(f"SHA mismatch for {name}")
                            os.replace(partial, output)
                            break
                        offset = partial.stat().st_size if partial.exists() else 0
                        headers = {"Range": f"bytes={offset}-"} if offset else {}
                        with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as transfer:
                            transfer.raise_for_status()
                            mode = "ab" if transfer.status_code == 206 and offset else "wb"
                            with partial.open(mode) as stream:
                                for chunk in transfer.iter_content(4 * 1024 * 1024):
                                    stream.write(chunk)
                        if partial.stat().st_size != size:
                            raise ValueError(f"Size mismatch for {name}")
                        if expected and digest(partial) != expected:
                            partial.unlink()
                            raise ValueError(f"SHA mismatch for {name}")
                        os.replace(partial, output)
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 4:
                            raise
                        time.sleep(3)
            manifest["files"].append({"path": name, "size": size, "sha256": digest(output), "lfs_sha256": expected, "source": url})
            print(repo, name, size, "verified", flush=True)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
