import os
import unittest
from unittest.mock import Mock, patch
import requests


def response(status=200, payload=None):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    return r


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "offline-test-only"}, clear=True
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_search_normalizes_results_and_uses_deepseek_endpoint(self):
        from agentD.providers.deepseek_search import DeepSeekSearch

        session = Mock()
        session.post.return_value = response(
            payload={
                "results": [
                    {
                        "title": "Paper",
                        "url": "https://example.org/paper",
                        "summary": "Evidence",
                    }
                ]
            }
        )
        service = DeepSeekSearch(session=session)
        hits = service.search("BCL-2 inhibitor")
        self.assertEqual(hits[0]["summary"], "Evidence")
        self.assertEqual(hits[0]["url"], "https://example.org/paper")
        self.assertEqual(
            session.post.call_args.args[0], "https://api.deepseek.com/v1/tools/search"
        )
        self.assertEqual(
            session.post.call_args.kwargs["headers"]["Authorization"],
            "Bearer offline-test-only",
        )

    def test_empty_results_are_not_invented(self):
        from agentD.providers.deepseek_search import DeepSeekSearch

        s = Mock()
        s.post.return_value = response(payload={"results": []})
        self.assertEqual(DeepSeekSearch(session=s).search("missing"), [])

    def test_bad_auth_or_balance_is_not_retried(self):
        from agentD.providers.deepseek_search import DeepSeekSearch, ServiceError

        for status in (401, 402):
            with self.subTest(status=status):
                s = Mock()
                s.post.return_value = response(status)
                with self.assertRaises(ServiceError) as e:
                    DeepSeekSearch(session=s).search("test")
                self.assertNotIn("offline-test-only", str(e.exception))
                self.assertEqual(s.post.call_count, 1)

    def test_transient_and_timeout_failures_have_bounded_retries(self):
        from agentD.providers.deepseek_search import DeepSeekSearch

        s = Mock()
        s.post.side_effect = [
            requests.Timeout(),
            response(429),
            response(payload={"results": []}),
        ]
        self.assertEqual(
            DeepSeekSearch(session=s, sleep=lambda _: None).search("test"), []
        )
        self.assertEqual(s.post.call_count, 3)

    def test_malformed_payload_or_query_fails_before_downstream_use(self):
        from agentD.providers.deepseek_search import DeepSeekSearch, ServiceError

        s = Mock()
        s.post.return_value = response(payload={"results": "not a list"})
        service = DeepSeekSearch(session=s)
        with self.assertRaises(ServiceError):
            service.search("test")
        for query in ("", "x" * 129):
            with self.assertRaises(ValueError):
                service.search(query)
        s.post.return_value.json.side_effect = ValueError("bad JSON")
        with self.assertRaises(ServiceError):
            service.search("test")

    def test_fetch_preserves_actual_content(self):
        from agentD.providers.deepseek_search import DeepSeekSearch

        s = Mock()
        s.post.return_value = response(
            payload={"title": "Paper", "content": "# Evidence"}
        )
        page = DeepSeekSearch(session=s).fetch("https://example.org/paper")
        self.assertEqual(page["content"], "# Evidence")
        with self.assertRaises(ValueError):
            DeepSeekSearch(session=s).fetch("file:///secret")
