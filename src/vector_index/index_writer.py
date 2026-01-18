from typing import List, Tuple, Dict
from vector_index.embedding import EmbeddingGenerator
from vector_index.index_registry import IndexRegistry


class VectorIndexWriter:
    """
    Writes chunked legal data into vector indexes.
    """

    def __init__(self, registry: IndexRegistry):
        self.embedder = EmbeddingGenerator()
        self.registry = registry

    def write(
        self,
        chunks: List[Tuple[str, Dict]]
    ):
        texts = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]

        embeddings = self.embedder.embed(texts)

        ids = [
            f"{m['source']}::{m['section_or_clause']}"
            for m in metadatas
        ]

        for text, meta, emb, record_id in zip(texts, metadatas, embeddings, ids):
            index = self._select_index(meta)
            index.add(
                ids=[record_id],
                embeddings=[emb],
                metadatas=[meta],
                texts=[text]
            )

        self.registry.legal_index.persist()
        self.registry.contract_index.persist()

    def _select_index(self, metadata: Dict):
        if metadata["doc_type"] in {"rera_act", "state_rule", "notification"}:
            return self.registry.legal_index
        return self.registry.contract_index
