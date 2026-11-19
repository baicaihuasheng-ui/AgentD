import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class OfflineIntegrationTests(unittest.TestCase):
    def test_pdf_rag_exposes_only_retrieved_source_and_page_to_model(self):
        from pypdf import PdfWriter
        from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_community.vectorstores import FAISS
        from agentD.documents import load_documents
        from agentD.local_embeddings import get_embeddings
        from agentD.utils import RetrievalQABypassTokenLimit

        captured = []

        class Capture(BaseCallbackHandler):
            def on_chat_model_start(self, serialized, messages, **kwargs):
                captured.extend(m.content for batch in messages for m in batch)

        with tempfile.TemporaryDirectory() as d:
            for name, text in [
                ("cancer", "BCL-2 inhibitors induce apoptosis in leukemia."),
                ("weather", "Weather forecasts predict rain and snow."),
            ]:
                writer = PdfWriter()
                page = writer.add_blank_page(width=612, height=792)
                font = DictionaryObject(
                    {
                        NameObject("/Type"): NameObject("/Font"),
                        NameObject("/Subtype"): NameObject("/Type1"),
                        NameObject("/BaseFont"): NameObject("/Helvetica"),
                    }
                )
                page[NameObject("/Resources")] = DictionaryObject(
                    {
                        NameObject("/Font"): DictionaryObject(
                            {NameObject("/F1"): writer._add_object(font)}
                        )
                    }
                )
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
                page[NameObject("/Contents")] = writer._add_object(stream)
                p = Path(d) / (name + ".pdf")
                with p.open("wb") as f:
                    writer.write(f)
                p.with_suffix(".pdf.source.json").write_text(
                    json.dumps(
                        {"source": f"https://example.org/{name}", "content_kind": "pdf"}
                    )
                )
            store = FAISS.from_documents(load_documents(d), get_embeddings())
            llm = FakeListChatModel(
                responses=["BCL-2 inhibition promotes apoptosis."],
                callbacks=[Capture()],
            )
            answer = RetrievalQABypassTokenLimit(
                store,
                "BCL-2 leukemia apoptosis",
                llm,
                k=1,
                min_k=1,
                return_sources=True,
            )
            self.assertEqual(
                answer["sources"][0]["source"], "https://example.org/cancer"
            )
            self.assertEqual(len(answer["sources"]), 1)
            self.assertIn("https://example.org/cancer", "\n".join(captured))
            self.assertIn("Page: 1", "\n".join(captured))

    def test_subprocess_inherits_config_and_never_serializes_key(self):
        env = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "AGENTD_MODEL": "deepseek-flash",
            "DEEPSEEK_API_KEY": "offline-test-only",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-s",
                "-c",
                "from agentD.runtime_config import load_runtime; c=load_runtime(); print(c.model); print(repr(c))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("deepseek-flash", result.stdout)
        self.assertNotIn("offline-test-only", result.stdout)

    def test_local_markdown_rag_uses_actual_source_without_a_live_key(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        import agentD.tools.retrieval as retrieval

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "paper.md"
            p.write_text(
                "BCL-2 inhibitors promote apoptosis in leukemia.", encoding="utf-8"
            )
            p.with_suffix(".md.source.json").write_text(
                json.dumps(
                    {
                        "source": "https://example.org/evidence",
                        "content_kind": "web_content",
                    }
                )
            )
            with (
                patch.object(retrieval, "PAPER_DIR", d),
                patch.object(
                    retrieval,
                    "get_chat_model",
                    return_value=FakeListChatModel(
                        responses=["BCL-2 inhibition promotes apoptosis."]
                    ),
                ),
            ):
                result = json.loads(
                    retrieval.question_answering.invoke(
                        "How does BCL-2 inhibition work?"
                    )
                )
            self.assertIn("apoptosis", result["answer"])
            self.assertEqual(
                result["available_sources"], ["https://example.org/evidence"]
            )

    def test_mcp_server_lists_tools_without_real_credentials(self):
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        root = Path(__file__).resolve().parents[1]

        async def exercise():
            transport = StdioTransport(
                command=sys.executable,
                args=[str(root / "mcp_agent.py")],
                cwd=str(root),
                env={
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
            async with Client(transport, timeout=60) as client:
                return {t.name for t in await client.list_tools()}

        names = asyncio.run(exercise())
        self.assertTrue({"agentd_run_pipeline", "agentd_qna"}.issubset(names))
