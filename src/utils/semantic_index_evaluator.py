from typing import Any, Dict, List, Optional

from utils.statute_normalizer import (
    normalize_section_ref,
    normalize_statutory_basis,
)


class SemanticIndexEvaluator:
    """
    Lawyer-aligned retrieval quality evaluator.

    Measures:
    - Coverage: statutory retrieval present when expected
    - Anchor Match: expected statute sections matched in retrieved evidence
    - Noise Penalty: proportion of likely-irrelevant evidence
    - Chunk Confidence: reliability of clause chunking/structure
    """

    STATUTORY_DOC_TYPES = {"rera_act", "state_rule"}

    def evaluate(
        self,
        clause_result: Any,
        evidence_pack: Any,
        chunk: Optional[Any] = None,
    ) -> Dict[str, Any]:
        basis = normalize_statutory_basis(
            getattr(clause_result, "statutory_basis", None)
        )

        expected_sections = basis.get("sections", []) if basis else []
        expected_set = set(expected_sections)
        intent = getattr(clause_result, "intent", "unknown")
        evidences = getattr(evidence_pack, "evidences", []) or []

        should_have_statutory = intent != "unknown" and bool(expected_sections)
        statutory_evidences = [
            ev for ev in evidences
            if (getattr(ev, "metadata", None) and ev.metadata.doc_type in self.STATUTORY_DOC_TYPES)
        ]
        coverage_ok = (not should_have_statutory) or bool(statutory_evidences)

        matched_sections: List[str] = []
        relevant_hits = 0
        for ev in evidences:
            candidates = self._evidence_section_candidates(ev)
            normalized_candidates = {
                c for c in (normalize_section_ref(x) for x in candidates) if c
            }

            if expected_set and normalized_candidates.intersection(expected_set):
                relevant_hits += 1
                for sec in normalized_candidates.intersection(expected_set):
                    if sec not in matched_sections:
                        matched_sections.append(sec)

        anchor_match = (not expected_set) or bool(matched_sections)

        total_hits = len(evidences)
        if total_hits == 0:
            noise_penalty = 1.0 if should_have_statutory else 0.0
        elif expected_set:
            noise_penalty = round(max(0.0, 1 - (relevant_hits / total_hits)), 2)
        else:
            # No expected anchor -> doc-type based noise proxy
            legal_hits = sum(
                1 for ev in evidences
                if (getattr(ev, "metadata", None) and ev.metadata.doc_type in self.STATUTORY_DOC_TYPES)
            )
            noise_penalty = round(max(0.0, 1 - (legal_hits / total_hits)), 2)

        chunk_confidence = self._chunk_confidence(clause_result, chunk)

        groundedness_score = self._groundedness_score(
            coverage_ok=coverage_ok,
            anchor_match=anchor_match,
            noise_penalty=noise_penalty,
            chunk_confidence=chunk_confidence,
        )

        reasons = self._reasons(
            should_have_statutory=should_have_statutory,
            coverage_ok=coverage_ok,
            anchor_match=anchor_match,
            expected_sections=expected_sections,
            matched_sections=matched_sections,
            noise_penalty=noise_penalty,
            chunk_confidence=chunk_confidence,
        )

        return {
            "coverage_ok": coverage_ok,
            "anchor_match": anchor_match,
            "noise_penalty": noise_penalty,
            "chunk_confidence": chunk_confidence,
            "groundedness_score": groundedness_score,
            "expected_sections": expected_sections,
            "matched_sections": matched_sections,
            "total_evidence_hits": total_hits,
            "relevant_hits": relevant_hits,
            "reasons": reasons,
        }

    def _evidence_section_candidates(self, evidence: Any) -> List[str]:
        out: List[str] = []
        sec = getattr(evidence, "section_or_clause", None)
        if sec:
            out.append(str(sec))

        metadata = getattr(evidence, "metadata", None)
        if metadata:
            for key in ("section_or_clause", "section", "rule", "clause", "chunk_id"):
                value = getattr(metadata, key, None)
                if value:
                    out.append(str(value))
        return out

    def _chunk_confidence(self, clause_result: Any, chunk: Optional[Any]) -> float:
        if getattr(clause_result, "semantic_confidence", None) is not None:
            return float(clause_result.semantic_confidence)
        if chunk is not None and getattr(chunk, "semantic_confidence", None) is not None:
            return float(chunk.semantic_confidence)
        if chunk is not None and getattr(chunk, "confidence", None) is not None:
            return float(chunk.confidence)
        return 0.5

    def _groundedness_score(
        self,
        coverage_ok: bool,
        anchor_match: bool,
        noise_penalty: float,
        chunk_confidence: float,
    ) -> float:
        # Weighted for legal grounding over pure retrieval volume.
        score = (
            0.35 * (1.0 if coverage_ok else 0.0)
            + 0.35 * (1.0 if anchor_match else 0.0)
            + 0.15 * (1.0 - noise_penalty)
            + 0.15 * max(0.0, min(1.0, chunk_confidence))
        )
        return round(max(0.0, min(1.0, score)), 2)

    def _reasons(
        self,
        should_have_statutory: bool,
        coverage_ok: bool,
        anchor_match: bool,
        expected_sections: List[str],
        matched_sections: List[str],
        noise_penalty: float,
        chunk_confidence: float,
    ) -> List[str]:
        reasons: List[str] = []
        if should_have_statutory and not coverage_ok:
            reasons.append("Expected statutory retrieval but no statutory evidence was found.")
        if expected_sections and not anchor_match:
            reasons.append(
                f"Retrieved evidence did not match expected anchors: {', '.join(expected_sections)}."
            )
        if expected_sections and matched_sections:
            reasons.append(
                f"Matched statutory anchors: {', '.join(matched_sections)}."
            )
        if noise_penalty > 0.5:
            reasons.append("High retrieval noise detected (majority of hits are likely irrelevant).")
        if chunk_confidence < 0.5:
            reasons.append("Low chunk confidence may reduce legal interpretability.")
        return reasons
