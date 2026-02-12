from typing import List, Dict, Any
import numpy as np

from retrieval.reranking_agent import CrossEncoderReRankingAgent
from retrieval.semantic_index_evaluator import SemanticIndexEvaluator

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
    Deterministic retrieval layer with:
    - Embedding search
    - Cross-encoder reranking
    - Semantic grounding evaluation
    - Strict evidence schema
    """

    TOP_K = 20
    RERANK_TOP_K = 10

    def __init__(self, index_registry: IndexRegistry):
        self.index_registry = index_registry

        self.embedder = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2"
        )

        self.reranker = CrossEncoderReRankingAgent(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=self.RERANK_TOP_K
        )

        self.semantic_index_evaluator = SemanticIndexEvaluator()

    # =========================================================
    # Public API
    # =========================================================

    def retrieve(
        self,
        clause_result: ClauseUnderstandingResult,
        state: str,
        clause_text: str | None = None,
    ) -> EvidencePack:

        evidences: List[Evidence] = []
        indexes = self.index_registry.get_indexes(state)

        # -----------------------------------------------------
        # Execute each retrieval query
        # -----------------------------------------------------
        for query in clause_result.retrieval_queries:

            index_names = self._resolve_indexes(query, indexes)

            search_text = (
                query.get("query_text")
                or query.get("intent")
                or clause_result.intent
            )

            if clause_text:
                snippet = " ".join(clause_text.strip().split())[:400]
                search_text = f"{search_text} {snippet}"

            query_embedding = self.embedder.embed([search_text])[0]
            query_embedding = np.array(query_embedding, dtype="float32")

            for index_name in index_names:

                index = indexes[index_name]

                # -------------------------------------------------
                # 1️⃣ Vector search
                # -------------------------------------------------
                documents: List[IndexDocument] = index.search(
                    query_embedding=query_embedding,
                    top_k=self.TOP_K
                )

                # -------------------------------------------------
                # 2️⃣ Cross-encoder reranking
                # -------------------------------------------------
                reranked_docs = self.reranker.rerank(
                    query=search_text,
                    documents=documents
                )[: self.RERANK_TOP_K]

                # -------------------------------------------------
                # 3️⃣ Apply metadata filters FIRST
                # -------------------------------------------------
                filtered_docs = [
                    doc for doc in reranked_docs
                    if self._passes_filters(
                        doc.metadata,
                        query.get("filters", {}),
                        state,
                        index_name
                    )
                ]

                # -------------------------------------------------
                # 4️⃣ Convert to Evidence models
                # -------------------------------------------------
                for doc in filtered_docs:

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

        # -----------------------------------------------------
        # 5️⃣ Grounding diagnostics (evaluate AFTER filtering)
        # -----------------------------------------------------
        diagnostics = self.semantic_index_evaluator.evaluate(
            clause_result=clause_result,
            evidence_docs=[
                {
                    "text": ev.text,
                    "metadata": ev.metadata.model_dump()
                }
                for ev in evidences
            ]
        )

        # -----------------------------------------------------
        # 6️⃣ Resolution
        # -----------------------------------------------------
        resolution = self._resolve_evidence(
            clause_result=clause_result,
            evidences=evidences
        )

        # -----------------------------------------------------
        # 7️⃣ Build EvidencePack
        # -----------------------------------------------------
        pack_data = {
            "clause_id": clause_result.clause_id,
            "clause_text": clause_text or "",
            "risk_level": clause_result.risk_level,
            "evidences": evidences,
            "resolution": resolution,
            "diagnostics": diagnostics,   # 🔥 attached
        }

        return build_model(
            EvidencePack,
            pack_data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    # =========================================================
    # Evidence resolution logic
    # =========================================================

    def _resolve_evidence(
        self,
        clause_result: ClauseUnderstandingResult,
        evidences: List[Evidence]
    ) -> str:

        if not evidences:
            return "INSUFFICIENT"

        for ev in evidences:
            doc_type = ev.metadata.doc_type
            if doc_type in {"model_agreement", "rera_act", "state_rule"}:
                return "EXPLICIT_ALIGNMENT"

        return "IMPLIED_ALIGNMENT"

    # =========================================================
    # Helpers
    # =========================================================

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