import tempfile
import unittest
from pathlib import Path


class DocumentTests(unittest.TestCase):
    def test_save_and_load_markdown_keeps_source_and_summary_label(self):
        from agentD.documents import collect_documents, load_documents

        class Search:
            def search(self, query, count):
                return [
                    {
                        "title": "Evidence",
                        "url": "https://example.org/paper",
                        "content": "",
                        "summary": "Reported findings",
                    }
                ]

            def fetch(self, url):
                raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as d:
            paths = collect_documents("BCL-2", Path(d), Search(), max_papers=1)
            docs = load_documents(Path(d))
            self.assertEqual(len(paths), 1)
            self.assertIn("Reported findings", docs[0].page_content)
            self.assertEqual(docs[0].metadata["source"], "https://example.org/paper")
            self.assertEqual(docs[0].metadata["content_kind"], "summary")

    def test_fetch_is_bounded_and_empty_results_create_no_fake_papers(self):
        from agentD.documents import collect_documents

        class Search:
            calls = 0

            def search(self, query, count):
                return [
                    {
                        "title": str(i),
                        "url": f"https://example.org/{i}",
                        "content": "",
                        "summary": "",
                    }
                    for i in range(5)
                ]

            def fetch(self, url):
                self.calls += 1
                raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as d:
            service = Search()
            self.assertEqual(
                collect_documents("query", Path(d), service, max_papers=5), []
            )
            self.assertEqual(service.calls, 3)
