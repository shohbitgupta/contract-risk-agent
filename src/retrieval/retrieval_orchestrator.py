from typing import List, Dict
import numpy as np

from RAG.models import (
    ClauseUnderstandingResult,
    Evidence,
    EvidencePack
)

from vector_index.embedding import EmbeddingGenerator
from vector_index.index_registry import IndexRegistry
from vector_index.index_base import IndexDocument


class RetrievalOrchestrator:
    """
    Deterministic retrieval layer.

    Responsibilities:
    - Convert clause intent into vector queries
    - Query appropriate legal vector indexes
    - Apply strict metadata filters
    - Return raw legal evidence (no reasoning, no LLM)
    """

    TOP_K = 5

    def __init__(self, index_registry: IndexRegistry):
        self.index_registry = index_registry
        self.embedder = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2"
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def retrieve(
        self,
        clause_result: ClauseUnderstandingResult,
        state: str
    ) -> EvidencePack:

        evidences: List[Evidence] = []

        # Load all indexes for the state (cached)
        indexes = self.index_registry.get_indexes(state)

        for query in clause_result.retrieval_queries:
            index_names = self._resolve_indexes(query, indexes)

            # Embed retrieval intent
            query_embedding = self.embedder.embed(
                [query["intent"]]
            )[0]
            query_embedding = np.array(
                query_embedding, dtype="float32"
            )

            for index_name in index_names:
                index = indexes[index_name]

                # Vector search (encapsulated)
                documents: List[IndexDocument] = index.search(
                    query_embedding=query_embedding,
                    top_k=self.TOP_K
                )

                for doc in documents:
                    if not self._passes_filters(
                        doc.metadata,
                        query.get("filters", {}),
                        state
                    ):
                        continue

                    evidences.append(
                        Evidence(
                            source=doc.metadata.get("source"),
                            section_or_clause=(
                                doc.metadata.get("section")
                                or doc.metadata.get("rule")
                                or doc.metadata.get("clause")
                                or doc.metadata.get("chunk_id")
                            ),
                            text=doc.content,
                            metadata=doc.metadata
                        )
                    )

        return EvidencePack(
            clause_id=clause_result.clause_id,
            intent=clause_result.intent,
            evidences=evidences
        )

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _resolve_indexes(
        self,
        query: Dict,
        indexes: Dict[str, object]
    ) -> List[str]:
        """
        Decide which indexes to query based on intent.
        """
        requested = query.get("index")

        if requested:
            if requested not in indexes:
                raise ValueError(
                    f"Requested index '{requested}' not available. "
                    f"Available: {list(indexes.keys())}"
                )
            return [requested]

        # Default: search all legal authority indexes
        return list(indexes.keys())

    def _passes_filters(
        self,
        metadata: dict,
        filters: dict,
        state: str
    ) -> bool:
        """
        Hard filters to prevent legal contamination.
        """

        # State filter (central laws may omit state)
        meta_state = metadata.get("state")
        if meta_state and meta_state != state:
            return False

        # Document type filter (optional)
        allowed_doc_types = filters.get("doc_type")
        if allowed_doc_types:
            if metadata.get("doc_type") not in allowed_doc_types:
                return False

        return True
