# src/retrieval/retrieval_orchestrator.py

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import math
import re
import numpy as np

from retrieval.reranking_agent import CrossEncoderReRankingAgent

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
    Statute-first, reranked, grounded retrieval layer.

    Improvements:
    - Statute bias in index priority
    - Better query construction
    - Cross-encoder reranking
    - Semantic quality evaluation (SIQ)
    """

    TOP_K = 20
    FINAL_TOP_K = 8

    # Hybrid retrieval: BM25 preselection before cross-encoder
    HYBRID_BM25_ENABLED = True
    BM25_PRESELECT_K = 60           # how many vector candidates to keep for reranking
    BM25_K1 = 1.6
    BM25_B = 0.75

    # =========================================================
    # Init
    # =========================================================

    def __init__(self, index_registry: IndexRegistry):
        self.index_registry = index_registry

        self.embedder = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2"
        )

        self.reranker = CrossEncoderReRankingAgent(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    # =========================================================
    # Public API
    # =========================================================

    def retrieve(
        self,
        clause_result: ClauseUnderstandingResult,
        state: str,
        clause_text: Optional[str] = None,
    ) -> EvidencePack:

        indexes = self.index_registry.get_indexes(state)
        evidences: List[Evidence] = []

        # -------------------------------------------------
        # Build optimized search text
        # -------------------------------------------------
        search_text = self._build_search_text(
            clause_result=clause_result,
            clause_text=clause_text
        )

        query_embedding = self.embedder.embed([search_text])[0]
        query_embedding = np.array(query_embedding, dtype="float32")

        # -------------------------------------------------
        # Statute-first index resolution
        # -------------------------------------------------
        index_names = self._resolve_indexes(
            clause_result=clause_result,
            indexes=indexes
        )

        candidate_docs: List[IndexDocument] = []

        # -------------------------------------------------
        # Vector retrieval
        # -------------------------------------------------
        for index_name in index_names:

            index = indexes[index_name]

            documents = index.search(
                query_embedding=query_embedding,
                top_k=self.TOP_K
            )

            candidate_docs.extend(documents)

        # -------------------------------------------------
        # Hard anchor injection (expected Section/Rule -> ensure present)
        # -------------------------------------------------
        candidate_docs.extend(
            self._inject_expected_anchor_docs(
                clause_result=clause_result,
                indexes=indexes,
            )
        )

        # -------------------------------------------------
        # Hybrid BM25 pre-selection (vector -> BM25 -> rerank)
        # -------------------------------------------------
        if self.HYBRID_BM25_ENABLED and len(candidate_docs) > self.BM25_PRESELECT_K:
            candidate_docs = self._bm25_preselect(
                query=search_text,
                docs=candidate_docs,
                clause_result=clause_result,
                k=self.BM25_PRESELECT_K,
            )

        # -------------------------------------------------
        # Cross-encoder reranking
        # -------------------------------------------------
        reranked_docs = self.reranker.rerank(
            query=search_text,
            documents=candidate_docs
        )

        reranked_docs = reranked_docs[: self.FINAL_TOP_K]

        # -------------------------------------------------
        # Force include expected statute/rule anchors
        # -------------------------------------------------
        anchor_docs = self._inject_expected_anchor_docs(
            clause_result=clause_result,
            indexes=indexes,
        )
        reranked_docs = self._ensure_anchor_docs_in_final(
            docs=reranked_docs,
            anchor_docs=anchor_docs,
            k=self.FINAL_TOP_K,
        )

        # -------------------------------------------------
        # Diagnostics (coverage / anchor match)
        # -------------------------------------------------
        diagnostics = self._compute_diagnostics(
            clause_result=clause_result,
            docs=reranked_docs,
        )

        # -------------------------------------------------
        # Convert to STRICT Evidence models
        # -------------------------------------------------
        for doc in reranked_docs:

            normalized = normalize_chunk_metadata(
                raw=doc.metadata,
                index_name=doc.metadata.get("index_name"),
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
        # Determine resolution
        # -------------------------------------------------
        resolution = self._resolve_evidence(
            clause_result=clause_result,
            evidences=evidences,
            diagnostics=diagnostics,
        )

        pack_data = {
            "clause_id": clause_result.clause_id,
            "clause_text": clause_text or "",
            "risk_level": clause_result.risk_level,
            "evidences": evidences,
            "diagnostics": diagnostics,
            "resolution": resolution,
        }

        return build_model(
            EvidencePack,
            pack_data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    def _ensure_anchor_docs_in_final(
        self,
        *,
        docs: List[IndexDocument],
        anchor_docs: List[IndexDocument],
        k: int,
    ) -> List[IndexDocument]:
        """
        Cross-encoder reranking can still drop the exact statute/rule anchors.
        Ensure anchor docs are always present in the final pack (up to k total).
        """
        if not anchor_docs:
            return docs[:k]

        def cid(d: IndexDocument) -> str:
            return (
                (d.metadata or {}).get("chunk_id")
                or (d.metadata or {}).get("id")
                or d.content[:40]
            )

        out: List[IndexDocument] = []
        seen: set[str] = set()

        # Put anchors first (deduped)
        for d in anchor_docs:
            key = cid(d)
            if key in seen:
                continue
            out.append(d)
            seen.add(key)
            if len(out) >= k:
                return out

        # Fill remaining with reranked docs
        for d in docs:
            key = cid(d)
            if key in seen:
                continue
            out.append(d)
            seen.add(key)
            if len(out) >= k:
                break

        return out

    # =========================================================
    # Search Text Builder (VERY IMPORTANT)
    # =========================================================

    def _build_search_text(
        self,
        clause_result: ClauseUnderstandingResult,
        clause_text: Optional[str]
    ) -> str:

        intent = clause_result.intent or "unknown"

        statute = ""
        if clause_result.statutory_basis:
            sections = clause_result.statutory_basis.get("sections", [])
            rules = clause_result.statutory_basis.get("state_rules", []) or []
            statute = " ".join([*sections, *rules])

        snippet = ""
        if clause_text:
            snippet = " ".join(clause_text.split())[:400]

        return f"{intent} {statute} {snippet}"

    # =========================================================
    # Hybrid BM25 helpers (no external dependency)
    # =========================================================

    _TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+[a-zA-Z]*")
    _SECTION_BASE_RE = re.compile(r"section\s*(\d+)", re.IGNORECASE)

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall((text or "").lower())

    def _unique_docs(self, docs: List[IndexDocument]) -> List[IndexDocument]:
        """
        Deduplicate docs coming from multiple indexes.
        """
        seen: set[str] = set()
        out: List[IndexDocument] = []
        for d in docs:
            cid = (
                (d.metadata or {}).get("chunk_id")
                or (d.metadata or {}).get("id")
                or str(id(d))
            )
            if cid in seen:
                continue
            seen.add(cid)
            out.append(d)
        return out

    def _bm25_preselect(
        self,
        *,
        query: str,
        docs: List[IndexDocument],
        clause_result: ClauseUnderstandingResult,
        k: int,
    ) -> List[IndexDocument]:
        """
        BM25 ranks the *vector-retrieved candidate set* and keeps top-k.
        This improves lexical precision (e.g., "refund", "interest", "Section 18")
        before cross-encoder reranking.
        """
        docs = self._unique_docs(docs)
        if not docs:
            return docs

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return docs[:k]

        doc_tokens = [self._tokenize(d.content) for d in docs]
        scores = self._bm25_scores(
            query_tokens=q_tokens,
            doc_tokens=doc_tokens,
            k1=self.BM25_K1,
            b=self.BM25_B,
        )

        # Keep expected statute anchors in the pool even if BM25 is weak
        must_keep = self._expected_anchor_docs(clause_result, docs)

        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        picked: List[IndexDocument] = []
        picked_ids: set[str] = set()

        for d in must_keep:
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid not in picked_ids:
                picked.append(d)
                picked_ids.add(cid)
            if len(picked) >= k:
                return picked

        for i in ranked:
            d = docs[i]
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid in picked_ids:
                continue
            picked.append(d)
            if cid:
                picked_ids.add(cid)
            if len(picked) >= k:
                break

        return picked

    def _expected_anchor_docs(
        self,
        clause_result: ClauseUnderstandingResult,
        docs: List[IndexDocument],
    ) -> List[IndexDocument]:
        """
        If ClauseUnderstanding predicted expected statute sections,
        keep any candidate docs whose metadata section matches.
        """
        basis = clause_result.statutory_basis or {}
        expected = [*(basis.get("sections") or []), *(basis.get("state_rules") or [])]
        if not expected:
            return []

        expected_norm = {str(s).lower().strip() for s in expected if s}
        expected_section_bases = {
            self._section_base(s) for s in (basis.get("sections") or []) if self._section_base(s)
        }
        out: List[IndexDocument] = []
        for d in docs:
            sec = (d.metadata or {}).get("section")
            if sec and str(sec).lower().strip() in expected_norm:
                out.append(d)
                continue
            # base-number match: expected Section 19(4) vs doc "Section 19"
            if expected_section_bases and sec:
                base = self._section_base(str(sec))
                if base and base in expected_section_bases:
                    out.append(d)
        return out

    def _inject_expected_anchor_docs(
        self,
        *,
        clause_result: ClauseUnderstandingResult,
        indexes: Dict[str, object],
    ) -> List[IndexDocument]:
        """
        If the clause has explicit expected anchors (sections/rules),
        ensure those exact statute/rule documents are present in candidates.

        This prevents model-agreement text from drowning out the statutory anchor.
        """
        basis = clause_result.statutory_basis or {}
        expected_sections = basis.get("sections") or []
        expected_rules = basis.get("state_rules") or []

        injected: List[IndexDocument] = []

        # ---- Sections (RERA Act) ----
        act_index = indexes.get("rera_act")
        if act_index is not None and expected_sections:
            for ref in expected_sections:
                sec_id = self._section_id_from_ref(str(ref))
                if not sec_id:
                    continue
                chunk_id = f"rera_act::section_{sec_id}"
                doc = getattr(act_index, "documents", {}).get(chunk_id)
                if doc is not None:
                    injected.append(doc)

        # ---- Rules (UP RERA Rules) ----
        rules_index = indexes.get("rera_rules")
        if rules_index is not None and expected_rules:
            expected_rules_norm = {self._normalize_rule_ref(str(r)) for r in expected_rules}
            expected_rules_norm.discard(None)
            if expected_rules_norm:
                for doc in getattr(rules_index, "documents", {}).values():
                    sec = (doc.metadata or {}).get("section")
                    sec_norm = self._normalize_rule_ref(str(sec)) if sec else None
                    if sec_norm and sec_norm in expected_rules_norm:
                        injected.append(doc)

        return injected

    def _section_id_from_ref(self, ref: str) -> Optional[str]:
        """
        "Section 19(1)" -> "19"
        "Section 18" -> "18"
        """
        m = re.search(r"(\d+)", ref or "", flags=re.IGNORECASE)
        return m.group(1) if m else None

    def _section_base(self, ref: str) -> Optional[str]:
        if not ref:
            return None
        s = str(ref).strip()
        if not s:
            return None
        m = re.search(r"section_(\d+)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        m = self._SECTION_BASE_RE.search(s)
        if m:
            return m.group(1)
        m = re.match(r"^\s*(\d+)", s)
        if m:
            return m.group(1)
        return None

    def _normalize_rule_ref(self, ref: str) -> Optional[str]:
        """
        Canonicalize rule refs like:
          "Rule 6" / "rule6" / "Rule 6(1)" -> "rule 6"
        """
        if not ref:
            return None
        s = " ".join(str(ref).split()).strip().lower()
        if not s:
            return None
        m = re.search(r"rule\s*(\d+)", s)
        if not m:
            return None
        return f"rule {m.group(1)}"

    def _bm25_scores(
        self,
        *,
        query_tokens: List[str],
        doc_tokens: List[List[str]],
        k1: float,
        b: float,
    ) -> List[float]:
        """
        Minimal BM25 implementation (Okapi BM25).
        """
        n_docs = len(doc_tokens)
        if n_docs == 0:
            return []

        doc_lens = [len(toks) for toks in doc_tokens]
        avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0

        # Document frequency
        df: Dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        # IDF with BM25+ style smoothing
        idf: Dict[str, float] = {}
        for t in set(query_tokens):
            dft = df.get(t, 0)
            idf[t] = math.log((n_docs - dft + 0.5) / (dft + 0.5) + 1.0)

        scores = [0.0 for _ in range(n_docs)]
        for i, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            dl = doc_lens[i] or 1
            denom_norm = k1 * (1.0 - b + b * (dl / (avgdl or 1.0)))

            s = 0.0
            for t in query_tokens:
                if t not in idf:
                    continue
                f = tf.get(t, 0)
                if f == 0:
                    continue
                s += idf[t] * (f * (k1 + 1.0)) / (f + denom_norm)
            scores[i] = s

        return scores

    # =========================================================
    # Statute-First Index Bias
    # =========================================================

    def _resolve_indexes(
        self,
        clause_result: ClauseUnderstandingResult,
        indexes: Dict[str, object]
    ) -> List[str]:

        priority = [
            "rera_act",
            "rera_rules",
            "model_bba",
            "case_law",
            "circulars"
        ]

        return [idx for idx in priority if idx in indexes]

    # =========================================================
    # Evidence Resolution
    # =========================================================

    def _resolve_evidence(
        self,
        clause_result: ClauseUnderstandingResult,
        evidences: List[Evidence],
        diagnostics: Dict[str, object],
    ) -> str:

        if not evidences:
            return "INSUFFICIENT"

        if bool(diagnostics.get("anchor_match")):
            return "EXPLICIT_ALIGNMENT"

        if bool(diagnostics.get("coverage")):
            return "IMPLIED_ALIGNMENT"

        return "INSUFFICIENT"

    def _compute_diagnostics(
        self,
        *,
        clause_result: ClauseUnderstandingResult,
        docs: List[IndexDocument],
    ) -> Dict[str, object]:
        """
        Produce EvidencePack diagnostics used downstream by LegalExplanationAgent.

        Kept intentionally simple and schema-compatible:
        - coverage: any evidence retrieved
        - anchor_match: expected statutory anchors found in evidence text/metadata
        - noise_ratio: fraction of evidence not matching expected anchors (when anchors exist)
        - groundedness: lightweight proxy for UI/debug
        """
        expected_sections: List[str] = []
        expected_rules: List[str] = []
        if clause_result.statutory_basis:
            expected_sections = clause_result.statutory_basis.get("sections", []) or []
            expected_rules = clause_result.statutory_basis.get("state_rules", []) or []

        coverage = len(docs) > 0

        expected_norm = [str(s).lower().strip() for s in expected_sections if s]
        expected_section_bases = [self._section_base(s) for s in expected_sections if self._section_base(s)]
        expected_rules_norm = [self._normalize_rule_ref(str(r)) for r in expected_rules if r]
        expected_rules_norm = [r for r in expected_rules_norm if r]
        anchor_match = False
        matching_docs = 0

        if (expected_norm or expected_rules_norm) and docs:
            for d in docs:
                text = (d.content or "").lower()
                sec_meta = str((d.metadata or {}).get("section") or "").lower().strip()
                hit = False
                # Sections
                for sec in expected_norm:
                    if sec and (sec in text or (sec_meta and sec == sec_meta)):
                        hit = True
                        anchor_match = True
                        break
                # Base section match: "Section 19(4)" should match "Section 19"
                if not hit and expected_section_bases and sec_meta:
                    base_meta = self._section_base(sec_meta)
                    if base_meta and base_meta in expected_section_bases:
                        hit = True
                        anchor_match = True
                # Rules
                if not hit and expected_rules_norm:
                    sec_rule_norm = self._normalize_rule_ref(sec_meta) if sec_meta else None
                    for r in expected_rules_norm:
                        if r and (
                            r in text
                            or (sec_rule_norm and r == sec_rule_norm)
                        ):
                            hit = True
                            anchor_match = True
                            break
                if hit:
                    matching_docs += 1

        if (expected_norm or expected_rules_norm) and docs:
            noise_ratio = 1 - (matching_docs / len(docs))
        else:
            noise_ratio = 0.0 if docs else 1.0

        groundedness = 1.0 if anchor_match else (0.6 if coverage else 0.0)

        return {
            "coverage": coverage,
            "anchor_match": anchor_match,
            "noise_ratio": round(float(noise_ratio), 2),
            "groundedness": float(groundedness),
            "expected_sections": expected_sections,
            "expected_rules": expected_rules,
        }
