import re
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
    _SECTION_BASE_RE = re.compile(r"section\s*(\d+)", re.IGNORECASE)

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
        expected_rules = basis.get("state_rules", []) if (basis and basis.get("state_rules")) else []
        expected_sections_set = set(expected_sections)
        expected_section_bases = {
            self._section_base(s) for s in expected_sections if self._section_base(s)
        }
        expected_rules_set = {self._normalize_rule_ref(r) for r in expected_rules if self._normalize_rule_ref(r)}
        intent = getattr(clause_result, "intent", "unknown")
        evidences = getattr(evidence_pack, "evidences", []) or []

        should_have_statutory = intent != "unknown" and bool(expected_sections or expected_rules)
        statutory_evidences = [
            ev for ev in evidences
            if (getattr(ev, "metadata", None) and ev.metadata.doc_type in self.STATUTORY_DOC_TYPES)
        ]
        coverage_ok = (not should_have_statutory) or bool(statutory_evidences)

        matched_sections: List[str] = []
        matched_rules: List[str] = []
        relevant_hits = 0
        for ev in evidences:
            candidates = self._evidence_section_candidates(ev)
            normalized_sections = {
                c for c in (normalize_section_ref(x) for x in candidates) if c
            }
            normalized_rules = {
                r for r in (self._normalize_rule_ref(x) for x in candidates) if r
            }
            section_bases = {
                self._section_base(x) for x in candidates if self._section_base(x)
            }

            hit = False
            # Section match: exact normalized, OR base-section-number match (e.g., Section 19(4) -> Section 19)
            if expected_sections_set and (
                normalized_sections.intersection(expected_sections_set)
                or (expected_section_bases and section_bases.intersection(expected_section_bases))
            ):
                hit = True
                exact_hits = normalized_sections.intersection(expected_sections_set)
                if exact_hits:
                    for sec in exact_hits:
                        if sec not in matched_sections:
                            matched_sections.append(sec)
                else:
                    # Record base matches using canonical "Section N"
                    for base in sorted(section_bases.intersection(expected_section_bases)):
                        sec = f"Section {base}"
                        if sec not in matched_sections:
                            matched_sections.append(sec)

            if expected_rules_set and normalized_rules.intersection(expected_rules_set):
                hit = True
                for r in normalized_rules.intersection(expected_rules_set):
                    if r not in matched_rules:
                        matched_rules.append(r)

            if hit:
                relevant_hits += 1

        anchor_match = (not expected_sections_set and not expected_rules_set) or bool(matched_sections or matched_rules)

        total_hits = len(evidences)
        if total_hits == 0:
            noise_penalty = 1.0 if should_have_statutory else 0.0
        elif expected_sections_set or expected_rules_set:
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
            "expected_rules": expected_rules,
            "matched_rules": matched_rules,
            "total_evidence_hits": total_hits,
            "relevant_hits": relevant_hits,
            "reasons": reasons,
        }

    def _normalize_rule_ref(self, ref: str) -> Optional[str]:
        if not ref:
            return None
        s = " ".join(str(ref).split()).strip().lower()
        m = re.search(r"rule\s*(\d+)", s)
        if not m:
            return None
        return f"rule {m.group(1)}"

    def _section_base(self, ref: str) -> Optional[str]:
        """
        Returns base section number as string:
          "Section 19(4)" -> "19"
          "19(4)" -> "19"
          "rera_act::section_19" -> "19"
        """
        if not ref:
            return None
        s = str(ref).strip()
        if not s:
            return None
        # chunk_id style
        m = re.search(r"section_(\d+)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        # "Section 19(4)" / "19(4)"
        m = self._SECTION_BASE_RE.search(s)
        if m:
            return m.group(1)
        m = re.match(r"^\s*(\d+)", s)
        if m:
            return m.group(1)
        return None

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
