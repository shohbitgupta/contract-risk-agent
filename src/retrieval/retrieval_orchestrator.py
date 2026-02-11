# from typing import List, Dict
# import numpy as np

# from RAG.models import (
#     ClauseUnderstandingResult,
#     Evidence,
#     EvidencePack
# )

# from vector_index.embedding import EmbeddingGenerator
# from vector_index.index_registry import IndexRegistry
# from vector_index.index_base import IndexDocument


# class RetrievalOrchestrator:
#     """
#     Deterministic retrieval layer.

#     Responsibilities:
#     - Convert clause intent into vector queries
#     - Query appropriate legal vector indexes
#     - Apply strict metadata filters
#     - Return raw legal evidence (no reasoning, no LLM)

#     Example:
#         >>> orchestrator = RetrievalOrchestrator(IndexRegistry(Path("src/data/vector_indexes"), 384))
#         >>> orchestrator.retrieve(clause_result, "uttar_pradesh")
#         EvidencePack(...)
#     """

#     TOP_K = 5

#     def __init__(self, index_registry: IndexRegistry):
#         self.index_registry = index_registry
#         self.embedder = EmbeddingGenerator(
#             model_name="all-MiniLM-L6-v2"
#         )

#     # -------------------------------------------------
#     # Public API
#     # -------------------------------------------------

#     def retrieve(
#         self,
#         clause_result: ClauseUnderstandingResult,
#         state: str
#     ) -> EvidencePack:
#         """
#         Retrieve legal evidence for a clause intent.

#         Returns:
#             EvidencePack with matched evidence chunks.
#         """

#         evidences: List[Evidence] = []

#         # Load all indexes for the state (cached)
#         indexes = self.index_registry.get_indexes(state)

#         for query in clause_result.retrieval_queries:
#             index_names = self._resolve_indexes(query, indexes)

#             # Embed retrieval intent
#             query_embedding = self.embedder.embed(
#                 [query["intent"]]
#             )[0]
#             query_embedding = np.array(
#                 query_embedding, dtype="float32"
#             )

#             for index_name in index_names:
#                 index = indexes[index_name]

#                 # Vector search (encapsulated)
#                 documents: List[IndexDocument] = index.search(
#                     query_embedding=query_embedding,
#                     top_k=self.TOP_K
#                 )

#                 for doc in documents:
#                     if not self._passes_filters(
#                         doc.metadata,
#                         query.get("filters", {}),
#                         state,
#                         index_name
#                     ):
#                         continue

#                     evidences.append(
#                         Evidence(
#                             source=doc.metadata.get("source"),
#                             section_or_clause=(
#                                 doc.metadata.get("section")
#                                 or doc.metadata.get("rule")
#                                 or doc.metadata.get("clause")
#                                 or doc.metadata.get("chunk_id")
#                             ),
#                             text=doc.content,
#                             metadata=doc.metadata
#                         )
#                     )

#         return EvidencePack(
#             clause_id=clause_result.clause_id,
#             intent=clause_result.intent,
#             evidences=evidences
#         )

#     # -------------------------------------------------
#     # Internal helpers
#     # -------------------------------------------------

#     def _resolve_indexes(
#         self,
#         query: Dict,
#         indexes: Dict[str, object]
#     ) -> List[str]:
#         """
#         Decide which indexes to query based on intent.

#         Example:
#             >>> self._resolve_indexes({"index": "rera_act"}, indexes)
#             ['rera_act']
#         """
#         requested = query.get("index")

#         if requested:
#             if requested not in indexes:
#                 raise ValueError(
#                     f"Requested index '{requested}' not available. "
#                     f"Available: {list(indexes.keys())}"
#                 )
#             return [requested]

#         # Default: search all legal authority indexes
#         return list(indexes.keys())

#     def _passes_filters(
#         self,
#         metadata: dict,
#         filters: dict,
#         state: str,
#         index_name: str
#     ) -> bool:
#         """
#         Hard filters to prevent legal contamination.

#         Returns:
#             True if metadata satisfies filters.
#         """

#         # State filter (central laws may omit state)
#         meta_state = metadata.get("state")
#         if meta_state and meta_state != state:
#             return False

#         # Document type filter (optional)
#         allowed_doc_types = filters.get("doc_type")
#         if allowed_doc_types:
#             inferred_doc_type = metadata.get("doc_type") or self._infer_doc_type(index_name)
#             if inferred_doc_type not in allowed_doc_types:
#                 return False

#         return True

#     def _infer_doc_type(self, index_name: str) -> str | None:
#         """
#         Infer doc_type from index name when metadata is missing.

#         Example:
#             >>> self._infer_doc_type("rera_rules")
#             'state_rule'
#         """
#         mapping = {
#             "rera_act": "rera_act",
#             "rera_rules": "state_rule",
#             "model_bba": "model_agreement",
#             "circulars": "notification",
#             "case_law": "case_law"
#         }
#         return mapping.get(index_name)

from typing import List, Dict
import numpy as np

from RAG.models import (
    ClauseUnderstandingResult,
    Evidence,
    EvidencePack,
    ChunkMetadata
)

from utils.schema_factory import build_model
from utils.schema_drift import log_schema_drift
from configs.schema_config import STRICT_SCHEMA

from vector_index.embedding import EmbeddingGenerator
from vector_index.index_registry import IndexRegistry
from vector_index.index_base import IndexDocument
from retrieval.metadata_normalizer import normalize_chunk_metadata


class RetrievalOrchestrator:
    """
    Deterministic retrieval layer with RERA compliance awareness.
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
        state: str,
        clause_text: str | None = None,
    ) -> EvidencePack:
        """
        Retrieve legal evidence for a clause intent.
        Uses semantic query (intent description + statutory sections + clause snippet)
        for better RERA relevance.
        """

        evidences: List[Evidence] = []
        indexes = self.index_registry.get_indexes(state)

        for query in clause_result.retrieval_queries:
            index_names = self._resolve_indexes(query, indexes)

            search_text = (
                query.get("query_text")
                or query.get("intent")
                or getattr(clause_result, "intent", "unknown")
            )
            if clause_text and clause_text.strip():
                snippet = " ".join(clause_text.strip().split())[:400]
                search_text = f"{search_text} {snippet}"

            query_embedding = self.embedder.embed([search_text])[0]
            query_embedding = np.array(query_embedding, dtype="float32")

            for index_name in index_names:
                index = indexes[index_name]

                documents: List[IndexDocument] = index.search(
                    query_embedding=query_embedding,
                    top_k=self.TOP_K
                )

                for doc in documents:
                    if not self._passes_filters(
                        doc.metadata,
                        query.get("filters", {}),
                        state,
                        index_name
                    ):
                        continue

                    # -----------------------------
                    # Metadata (STRICT)
                    # -----------------------------
                    normalized = normalize_chunk_metadata(
                        raw=doc.metadata,
                        index_name=index_name,
                        state=state
                    )

                    metadata = build_model(
                        ChunkMetadata,
                        normalized,
                        strict=STRICT_SCHEMA,
                        log_fn=log_schema_drift
                    )

                    evidence_data = {
                        "source": metadata.source,
                        "section_or_clause": (
                            doc.metadata.get("section")
                            or doc.metadata.get("rule")
                            or doc.metadata.get("clause")
                            or doc.metadata.get("chunk_id")
                        ),
                        "text": doc.content,
                        "metadata": metadata,
                    }

                    evidences.append(
                        build_model(
                            Evidence,
                            evidence_data,
                            strict=STRICT_SCHEMA,
                            log_fn=log_schema_drift
                        )
                    )

        # -------------------------------------------------
        # Compute resolution
        # -------------------------------------------------
        resolution = self._resolve_evidence(
            clause_result=clause_result,
            evidences=evidences
        )

        pack_data = {
            "clause_id": clause_result.clause_id,
            "clause_text": getattr(clause_result, "clause_text", ""),
            "risk_level": clause_result.risk_level,
            "evidences": evidences,
            "resolution": resolution,
        }

        return build_model(
            EvidencePack,
            pack_data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    # -------------------------------------------------
    # Evidence resolution logic
    # -------------------------------------------------

    def _resolve_evidence(
        self,
        clause_result: ClauseUnderstandingResult,
        evidences: List[Evidence]
    ) -> str:
        """
        Determine how evidence should be interpreted.
        """

        if not evidences:
            if getattr(clause_result, "compliance_mode", None) == "IMPLICIT":
                return "IMPLIED_ALIGNMENT"
            return "INSUFFICIENT"

        for ev in evidences:
            doc_type = ev.metadata.doc_type
            if doc_type in {"model_agreement", "rera_act", "state_rule"}:
                if getattr(clause_result, "compliance_mode", None) == "IMPLICIT":
                    return "IMPLIED_ALIGNMENT"
                return "EXPLICIT_ALIGNMENT"

        return "INSUFFICIENT"

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _resolve_indexes(
        self,
        query: Dict,
        indexes: Dict[str, object]
    ) -> List[str]:
        requested = query.get("index")
        if requested:
            if requested not in indexes:
                raise ValueError(
                    f"Requested index '{requested}' not available. "
                    f"Available: {list(indexes.keys())}"
                )
            return [requested]

        priority = [
            "model_bba",
            "rera_act",
            "rera_rules",
            "circulars",
            "case_law"
        ]

        return [idx for idx in priority if idx in indexes]

    def _passes_filters(
        self,
        metadata: dict,
        filters: dict,
        state: str,
        index_name: str
    ) -> bool:

        meta_state = metadata.get("state")
        if meta_state and meta_state != state:
            return False

        allowed_doc_types = filters.get("doc_type")
        if allowed_doc_types:
            inferred = metadata.get("doc_type") or self._infer_doc_type(index_name)
            if inferred not in allowed_doc_types:
                return False

        return True

    def _infer_doc_type(self, index_name: str) -> str | None:
        mapping = {
            "rera_act": "rera_act",
            "rera_rules": "state_rule",
            "model_bba": "model_agreement",
            "circulars": "notification",
            "case_law": "case_law"
        }
        return mapping.get(index_name)
