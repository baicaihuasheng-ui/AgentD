import os
import unittest
from unittest.mock import patch


class EmbeddingTests(unittest.TestCase):
    def test_missing_model_fails_without_network_download(self):
        from agentD.local_embeddings import LocalE5Embeddings

        with self.assertRaises(FileNotFoundError):
            LocalE5Embeddings("Z:/nonexistent-e5").embed_query("test")

    def test_empty_document_batch_does_not_load_weights(self):
        from agentD.local_embeddings import LocalE5Embeddings

        self.assertEqual(LocalE5Embeddings("Z:/nonexistent-e5").embed_documents([]), [])

    def test_real_model_vectors_are_normalized_and_retrievable_offline(self):
        import numpy as np
        import faiss
        from agentD.local_embeddings import get_embeddings

        with patch.dict(
            os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        ):
            embed = get_embeddings()
            vectors = np.asarray(
                embed.embed_documents(
                    [
                        "BCL-2 inhibitors induce apoptosis in leukemia cancer cells.",
                        "Weather forecasting predicts rain and snow.",
                    ]
                ),
                dtype="float32",
            )
            self.assertEqual(vectors.shape, (2, 384))
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            index = faiss.IndexFlatL2(embed.dimension)
            index.add(vectors)
            _, ids = index.search(
                np.asarray(
                    [embed.embed_query("白血病细胞的凋亡和 BCL-2 抑制剂")],
                    dtype="float32",
                ),
                1,
            )
            self.assertEqual(ids[0, 0], 0)

    def test_long_document_splits_with_source_preserved(self):
        from langchain_core.documents import Document
        from agentD.local_embeddings import split_documents, get_embeddings

        docs = split_documents(
            [
                Document(
                    page_content="BCL-2 apoptosis leukemia. " * 1000,
                    metadata={"source": "https://example.org/paper"},
                )
            ]
        )
        self.assertGreater(len(docs), 1)
        tokenizer = get_embeddings().tokenizer
        for doc in docs:
            self.assertEqual(doc.metadata["source"], "https://example.org/paper")
            self.assertLessEqual(
                len(tokenizer.encode("passage: " + doc.page_content)), 512
            )
