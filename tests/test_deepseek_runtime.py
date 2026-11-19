import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RuntimeTests(unittest.TestCase):
    def test_missing_deepseek_key_does_not_reuse_other_credentials(self):
        from agentD.runtime_config import RuntimeConfig

        with patch.dict(os.environ, {"OPENAI_API_KEY": "old-test-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                RuntimeConfig.from_env().require_key()

    def test_config_repr_does_not_expose_secret_and_resolves_paths(self):
        from agentD.runtime_config import RuntimeConfig

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "private-test-value"}, clear=True
        ):
            cfg = RuntimeConfig.from_env()
            self.assertNotIn("private-test-value", repr(cfg))
            self.assertTrue(cfg.embedding_path.is_absolute())

    def test_env_file_load_is_explicit_and_environment_has_priority(self):
        from agentD.runtime_config import load_runtime

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "credentials.env"
            f.write_text("DEEPSEEK_API_KEY=file-value\n")
            with patch.dict(
                os.environ,
                {"AGENTD_ENV_FILE": str(f), "DEEPSEEK_API_KEY": "env-value"},
                clear=True,
            ):
                self.assertEqual(load_runtime().require_key(), "env-value")

    def test_chat_request_targets_deepseek_and_disables_thinking(self):
        import httpx
        from agentD.providers.deepseek import get_chat_model

        captured = []

        def respond(request):
            import json

            captured.append((str(request.url), json.loads(request.content)))
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
                            "message": {
                                "role": "assistant",
                                "content": "offline answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "offline-test-only"}, clear=True
        ):
            llm = get_chat_model(
                http_client=httpx.Client(transport=httpx.MockTransport(respond))
            )
            self.assertEqual(llm.invoke("test").content, "offline answer")
        self.assertEqual(captured[0][0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured[0][1]["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
