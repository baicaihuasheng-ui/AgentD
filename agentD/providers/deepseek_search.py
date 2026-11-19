"""DeepSeek Search/Fetch with bounded retries and redacted errors."""

import time
from urllib.parse import urlparse
import requests
from agentD.runtime_config import load_runtime
from agentD.call_budget import current_budget


class ServiceError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def valid_web_url(url):
    p = urlparse(url)
    return (
        p.scheme in ("http", "https")
        and bool(p.hostname)
        and not p.username
        and not p.password
    )


class DeepSeekSearch:
    def __init__(self, session=None, sleep=time.sleep, max_attempts=3):
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.config = load_runtime()
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_attempts = max_attempts
        if current_budget():
            self.max_attempts = 1

    def _post(self, operation, payload):
        key = self.config.require_key()
        for attempt in range(self.max_attempts):
            budget = current_budget()
            token = budget.reserve(operation) if budget else None
            try:
                response = self.session.post(
                    self.config.api_base + "/v1/tools/" + operation,
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=(10, 60),
                    allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError):
                if budget:
                    budget.finish(token, success=False)
                if attempt == self.max_attempts - 1:
                    raise ServiceError(
                        f"DeepSeek {operation}: network timeout/connection failure"
                    ) from None
                self.sleep(0.5 * (2**attempt))
                continue
            status = response.status_code
            if budget:
                budget.finish(token, success=status == 200)
            if status == 200:
                try:
                    data = response.json()
                except ValueError:
                    raise ServiceError(f"DeepSeek {operation}: invalid JSON") from None
                if not isinstance(data, dict):
                    raise ServiceError(f"DeepSeek {operation}: invalid response object")
                return data
            if (
                status in (408, 429) or status >= 500
            ) and attempt < self.max_attempts - 1:
                self.sleep(0.5 * (2**attempt))
                continue
            raise ServiceError(
                f"DeepSeek {operation}: HTTP {status}", status_code=status
            )

    def search(self, query, count=5):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128:
            raise ValueError(
                "Search query must contain 1–128 characters; reformulate a shorter query"
            )
        if not 1 <= count <= 50:
            raise ValueError("Search count must be between 1 and 50")
        if current_budget():
            count = min(count, 3)
        data = self._post(
            "search", {"query": query.strip(), "count": count, "need_content": True}
        )
        results = data.get("results")
        if not isinstance(results, list):
            raise ServiceError("DeepSeek search: results must be a list")
        hits = []
        for item in results:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("url"), str)
                or not valid_web_url(item["url"])
            ):
                continue
            hits.append(
                {
                    k: item.get(k, "") if isinstance(item.get(k, ""), str) else ""
                    for k in ("title", "url", "snippet", "summary", "content")
                }
            )
        return hits

    def fetch(self, url):
        if not isinstance(url, str) or len(url) > 8192 or not valid_web_url(url):
            raise ValueError(
                "Fetch requires an HTTP(S) URL without embedded credentials"
            )
        data = self._post("fetch", {"url": url})
        if not isinstance(data.get("content"), str) or not data["content"].strip():
            raise ServiceError("DeepSeek fetch: no page content")
        return {
            "title": str(data.get("title") or ""),
            "content": data["content"],
            "url": url,
        }
