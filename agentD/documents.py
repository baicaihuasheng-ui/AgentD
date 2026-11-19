"""Source-preserving collection/loading of public PDFs and web text."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from agentD.providers.deepseek_search import valid_web_url


def _save_pdf(url, path):
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            if response.status_code != 200:
                return False
            size = 0
            with path.with_suffix(".part").open("wb") as out:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue
                    if size == 0 and not chunk.startswith(b"%PDF-"):
                        return False
                    size += len(chunk)
                    if size > 50 * 1024 * 1024:
                        return False
                    out.write(chunk)
            if not size:
                return False
            path.with_suffix(".part").replace(path)
            return True
    except requests.RequestException:
        return False
    finally:
        path.with_suffix(".part").unlink(missing_ok=True)


def collect_documents(query, directory, service, max_papers=10):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    seen = set()
    fetches = 0
    for hit in service.search(query, count=min(max_papers, 50)):
        url = hit.get("url", "")
        if not valid_web_url(url) or url in seen:
            continue
        seen.add(url)
        title = hit.get("title") or url
        stem = (
            re.sub(r"[^\w-]+", "_", title)[:60]
            + "_"
            + hashlib.sha256(url.encode()).hexdigest()[:12]
        )
        metadata = {"source": url, "title": title}
        path = directory / (stem + ".pdf")
        if urlparse(url).path.lower().endswith(".pdf") and _save_pdf(url, path):
            metadata["content_kind"] = "pdf"
        else:
            text = hit.get("content") or ""
            kind = "web_content"
            if not text and fetches < 3:
                fetches += 1
                try:
                    text = service.fetch(url)["content"]
                except (RuntimeError, ValueError, requests.RequestException):
                    pass
            if not text:
                text = hit.get("summary") or hit.get("snippet") or ""
                kind = "summary"
            if not text:
                continue
            metadata["content_kind"] = kind
            path = directory / (stem + ".md")
            path.write_text(
                f"# {title}\n\nSource: {url}\nContent kind: {kind}\n\n{text}",
                encoding="utf-8",
            )
        path.with_suffix(path.suffix + ".source.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        paths.append(str(path))
    return paths


def load_documents(directory):
    documents = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in (".pdf", ".md"):
            continue
        sidecar = path.with_suffix(path.suffix + ".source.json")
        metadata = (
            json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar.exists()
            else {"source": str(path), "content_kind": path.suffix[1:]}
        )
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            pages = PyPDFLoader(str(path)).load()
            for page in pages:
                page.metadata.update(metadata)
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            documents.extend(pages)
        else:
            metadata["page_number"] = "web"
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"), metadata=metadata
                )
            )
    return documents
