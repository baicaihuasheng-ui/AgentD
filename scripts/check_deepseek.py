"""Explicit, single-attempt live checks. The default performs no network requests."""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from agentD.runtime_config import load_runtime
from agentD.providers.deepseek import get_chat_model
from agentD.providers.deepseek_search import DeepSeekSearch, valid_web_url

MODEL = "deepseek-flash"
QUERY = "BCL-2 inhibitors site:pubmed.ncbi.nlm.nih.gov"


def run_check(live=False, report_path=None):
    report = {
        "mode": "live" if live else "configuration_only",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL,
        "thinking": "disabled",
        "max_output_tokens": 64,
        "automatic_retries": 0,
        "attempts": {"chat": 0, "search": 0, "fetch": 0},
        "steps": {},
    }

    def save():
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    operation = "configuration"
    try:
        cfg = load_runtime()
        report["credential_present"] = bool(cfg.api_key)
        if not cfg.api_key:
            report["status"] = "needs_key"
            save()
            return report
        if cfg.api_base != "https://api.deepseek.com":
            report["status"] = "blocked"
            report["reason"] = (
                "This bounded check only permits the official DeepSeek endpoint"
            )
            save()
            return report
        if not live:
            report["status"] = "ready"
            save()
            return report

        operation = "chat"
        report["status"] = "running"
        report["attempts"][operation] = 1
        save()
        with httpx.Client(follow_redirects=False) as client:
            llm = get_chat_model(
                model=MODEL, max_tokens=64, max_retries=0, http_client=client
            )
            message = llm.invoke("Reply with exactly OK and nothing else.")
        if not isinstance(message.content, str) or message.content.strip() != "OK":
            raise ValueError("Unexpected short-test response")
        usage = getattr(message, "usage_metadata", None) or {}
        report["steps"]["chat"] = {
            "status": "passed",
            "reply_matches": True,
            "usage": {
                key: usage[key]
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if isinstance(usage.get(key), int) and not isinstance(usage[key], bool)
            },
        }
        save()

        service = DeepSeekSearch(max_attempts=1)
        operation = "search"
        report["attempts"][operation] = 1
        save()
        hits = service.search(QUERY, count=3)
        safe_hits = []
        for hit in hits[:3]:
            url = hit.get("url", "")
            if not valid_web_url(url) or cfg.api_key in url:
                continue
            title = str(hit.get("title") or "").replace(cfg.api_key, "<redacted>")[:300]
            safe_hits.append({"title": title, "url": url})
        report["steps"]["search"] = {
            "status": "passed" if safe_hits else "empty",
            "query": QUERY,
            "result_count": len(safe_hits),
            "sources": safe_hits,
        }
        if not safe_hits:
            report["steps"]["fetch"] = {
                "status": "skipped",
                "reason": "No usable search result",
            }
            report["status"] = "partial"
            save()
            return report
        save()

        operation = "fetch"
        report["attempts"][operation] = 1
        save()
        page = service.fetch(safe_hits[0]["url"])
        content = page.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty page body")
        report["steps"]["fetch"] = {
            "status": "passed",
            "url": safe_hits[0]["url"],
            "content_characters": len(content),
        }
        report["status"] = "passed"
    except Exception as exc:
        # Provider errors may echo credentials. Never persist/print their messages,
        # response bodies, request objects, headers or exception tracebacks.
        status = getattr(exc, "status_code", None)
        report["status"] = "failed"
        report["steps"][operation] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "http_status": status if isinstance(status, int) else None,
        }
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send the three bounded paid requests, after cost confirmation",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Explicit local credential file; never print its contents",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.env_file:
        os.environ["AGENTD_ENV_FILE"] = str(args.env_file.expanduser().resolve())
    elif not os.getenv("AGENTD_ENV_FILE") and (ROOT / ".env.deepseek").is_file():
        os.environ["AGENTD_ENV_FILE"] = str(ROOT / ".env.deepseek")
    report_path = args.report or ROOT / "reports" / (
        "deepseek-check-"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        + ".json"
    )
    report = run_check(live=args.live, report_path=report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Report: " + str(report_path))
    return {"passed": 0, "ready": 0, "needs_key": 2, "partial": 3}.get(
        report["status"], 1
    )


if __name__ == "__main__":
    sys.exit(main())
