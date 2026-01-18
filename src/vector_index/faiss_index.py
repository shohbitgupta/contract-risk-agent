import faiss
import json
from pathlib import Path
from typing import List
import numpy as np

from vector_index.index_base import IndexDocument


class FAISSVectorIndex:
    """
    FAISS-based vector index with sidecar metadata storage.

    Stores:
    - FAISS index (embeddings only)
    - IndexDocument metadata + text (JSON)

    Acts as BOTH:
    - Index writer (offline ingestion)
    - Index reader (runtime retrieval)
    """

    def __init__(self, index_path: Path, dim: int):
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.json")

        # Exact inner product similarity
        # Use normalized embeddings → cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        # chunk_id → IndexDocument
        self.documents: dict[str, IndexDocument] = {}

    # -------------------------------------------------
    # Add documents (INGESTION TIME)
    # -------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[IndexDocument]
    ):
        if len(embeddings) != len(documents):
            raise ValueError(
                "Embeddings count does not match documents count"
            )

        # FAISS requires float32
        embeddings = embeddings.astype("float32")

        self.index.add(embeddings)

        for doc in documents:
            chunk_id = doc.metadata["chunk_id"]
            self.documents[chunk_id] = doc

    # -------------------------------------------------
    # Persist to disk
    # -------------------------------------------------

    def persist(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Write metadata sidecar
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    chunk_id: doc.to_dict()
                    for chunk_id, doc in self.documents.items()
                },
                f,
                indent=2,
                ensure_ascii=False
            )

    # -------------------------------------------------
    # Load from disk (RUNTIME)
    # -------------------------------------------------

    @classmethod
    def load(cls, index_path: Path, dim: int) -> "FAISSVectorIndex":
        obj = cls(index_path=index_path, dim=dim)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not obj.meta_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found: {obj.meta_path}"
            )

        obj.index = faiss.read_index(str(index_path))

        with open(obj.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for chunk_id, payload in raw.items():
            obj.documents[chunk_id] = IndexDocument(
                content=payload["content"],
                metadata=payload["metadata"]
            )

        return obj

    # -------------------------------------------------
    # Search (RUNTIME)
    # -------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[IndexDocument]:

        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        docs: List[IndexDocument] = []
        keys = list(self.documents.keys())

        for idx in indices[0]:
            if idx < 0 or idx >= len(keys):
                continue
            docs.append(self.documents[keys[idx]])

        return docs
