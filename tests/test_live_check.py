import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class LiveCheckTests(unittest.TestCase):
    def setUp(self):
        self.key = "offline-test-value-never-log"
        self.env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": self.key}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_check_never_constructs_network_clients(self):
        from scripts.check_deepseek import run_check

        with (
            patch("scripts.check_deepseek.get_chat_model") as chat,
            patch("scripts.check_deepseek.DeepSeekSearch") as search,
        ):
            report = run_check()
        self.assertEqual(report["status"], "ready")
        self.assertEqual(sum(report["attempts"].values()), 0)
        chat.assert_not_called()
        search.assert_not_called()
        self.assertNotIn(self.key, json.dumps(report))

    def test_live_check_limits_each_operation_and_saves_redacted_result(self):
        from scripts.check_deepseek import run_check

        chat = Mock()
        chat.invoke.return_value = SimpleNamespace(
            content="OK",
            usage_metadata={"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
        )
        search = Mock()
        search.search.return_value = [
            {"title": "BCL-2 study", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"}
        ]
        search.fetch.return_value = {"content": "# Study body", "title": "Study"}
        with tempfile.TemporaryDirectory() as d:
            report_path = Path(d) / "report.json"
            with (
                patch(
                    "scripts.check_deepseek.get_chat_model", return_value=chat
                ) as factory,
                patch(
                    "scripts.check_deepseek.DeepSeekSearch", return_value=search
                ) as search_factory,
            ):
                report = run_check(live=True, report_path=report_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["attempts"], {"chat": 1, "search": 1, "fetch": 1})
            self.assertEqual(factory.call_args.kwargs["max_tokens"], 64)
            self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
            self.assertEqual(search_factory.call_args.kwargs["max_attempts"], 1)
            self.assertEqual(search.search.call_args.kwargs["count"], 3)
            self.assertNotIn(self.key, report_path.read_text())
            self.assertNotIn("Authorization", report_path.read_text())

    def test_first_failure_stops_and_does_not_log_exception_message(self):
        from scripts.check_deepseek import run_check

        chat = Mock()
        chat.invoke.side_effect = RuntimeError("server echoed " + self.key)
        with (
            patch("scripts.check_deepseek.get_chat_model", return_value=chat),
            patch("scripts.check_deepseek.DeepSeekSearch") as search,
        ):
            report = run_check(live=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["attempts"], {"chat": 1, "search": 0, "fetch": 0})
        self.assertNotIn(self.key, json.dumps(report))
        search.assert_not_called()

    def test_no_search_results_skips_fetch_and_marks_partial(self):
        from scripts.check_deepseek import run_check

        chat = Mock()
        chat.invoke.return_value = SimpleNamespace(content="OK", usage_metadata={})
        search = Mock()
        search.search.return_value = []
        with (
            patch("scripts.check_deepseek.get_chat_model", return_value=chat),
            patch("scripts.check_deepseek.DeepSeekSearch", return_value=search),
        ):
            report = run_check(live=True)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["attempts"]["fetch"], 0)
        search.fetch.assert_not_called()

    def test_missing_key_and_nonofficial_endpoint_make_no_requests(self):
        from scripts.check_deepseek import run_check

        for env in (
            {"DEEPSEEK_API_KEY": ""},
            {"AGENTD_API_BASE": "https://example.org"},
        ):
            with (
                self.subTest(env=env),
                patch.dict(os.environ, env),
                patch("scripts.check_deepseek.get_chat_model") as chat,
            ):
                report = run_check(live=True)
                self.assertIn(report["status"], ("needs_key", "blocked"))
                self.assertEqual(sum(report["attempts"].values()), 0)
                chat.assert_not_called()

    def test_search_single_attempt_does_not_retry_429(self):
        from agentD.providers.deepseek_search import DeepSeekSearch, ServiceError

        session = Mock()
        session.post.return_value.status_code = 429
        with self.assertRaises(ServiceError):
            DeepSeekSearch(session=session, max_attempts=1).search("BCL-2")
        self.assertEqual(session.post.call_count, 1)

    def test_chat_factory_allows_the_small_token_limit(self):
        import httpx
        from agentD.providers.deepseek import get_chat_model

        seen = []

        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = get_chat_model(max_tokens=64, max_retries=0, http_client=client)
            self.assertEqual(llm.invoke("Reply OK").content, "OK")
        self.assertEqual(seen[0]["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", seen[0])
        self.assertEqual(seen[0]["thinking"], {"type": "disabled"})

    def test_chat_zero_retries_means_one_transport_attempt_on_429(self):
        import httpx
        from openai import RateLimitError
        from agentD.providers.deepseek import get_chat_model

        seen = []

        def respond(request):
            seen.append(request.url)
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "offline simulated limit",
                        "type": "rate_limit_error",
                    }
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = get_chat_model(max_tokens=64, max_retries=0, http_client=client)
            with self.assertRaises(RateLimitError):
                llm.invoke("Reply OK")
        self.assertEqual(len(seen), 1)
