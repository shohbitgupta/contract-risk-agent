from __future__ import annotations

from typing import List, Optional
from constants.alignment import ALLOWED_ALIGNMENTS

from RAG.contract_analysis import (
    ContractAnalysisResult,
    ContractSummary,
    ContractRiskDistribution,
    KeyIssue,
    ClauseAnalysisResult
)

from configs.callibration.callibration_config_loader import CalibrationConfig


RISK_RELEVANT_ROLES = {
    "obligation",
    "right",
    "procedure",
}


class ContractAggregationAgent:

    def __init__(self, calibration: CalibrationConfig):
        self.calibration = calibration

    # =========================================================
    # PUBLIC API
    # =========================================================

    def aggregate(
        self,
        clauses: List[ClauseAnalysisResult],
    ) -> ContractAnalysisResult:

        if not clauses:
            raise ValueError("Cannot aggregate empty clause list")

        raw_dist = {
            "aligned": 0,
            "partially_aligned": 0,
            "insufficient_evidence": 0,
            "contradiction": 0
        }

        risk_clauses: List[ClauseAnalysisResult] = []
        issues: List[KeyIssue] = []

        # -------------------------------------------------
        # Classify clauses
        # -------------------------------------------------
        for c in clauses:
            alignment = c.alignment
            if alignment == "conflicting":
                alignment = "contradiction"

            if alignment not in ALLOWED_ALIGNMENTS:
                raise ValueError(
                    f"Invalid alignment '{alignment}' for clause {c.clause_id}"
                )

            raw_dist[alignment] += 1

            if getattr(c, "clause_role", None) in RISK_RELEVANT_ROLES:
                risk_clauses.append(c)

        total_risk = len(risk_clauses) or 1

        risk_dist = {
            "aligned": sum(1 for c in risk_clauses if c.alignment == "aligned"),
            "partially_aligned": sum(1 for c in risk_clauses if c.alignment == "partially_aligned"),
            "insufficient_evidence": sum(1 for c in risk_clauses if c.alignment == "insufficient_evidence"),
            "contradiction": sum(1 for c in risk_clauses if c.alignment == "contradiction"),
        }

        insufficient_ratio = risk_dist["insufficient_evidence"] / total_risk
        partially_ratio = risk_dist["partially_aligned"] / total_risk

        # -------------------------------------------------
        # Top issues (lawyer-facing)
        # IMPORTANT: This does NOT change score/confidence.
        # -------------------------------------------------
        issues = self._build_top_issues(risk_clauses)

        # =========================================================
        # NEW CLEAN 3-FACTOR CONTRACT SCORE
        # =========================================================

        legal_core = self._legal_core_score(risk_clauses)
        clarity = self._clarity_score(risk_dist, total_risk)
        grounding = self._avg_grounding(risk_clauses)

        contract_score = round(
            0.5 * legal_core +
            0.3 * clarity +
            0.2 * grounding,
            2
        )

        # -------------------------------------------------
        # Semantic consistency caps (verdict ↔ score coherence)
        # -------------------------------------------------
        # If a large fraction of enforceable clauses are "insufficient_evidence",
        # the contract cannot be scored as "very safe" even if grounding is high.
        #
        # This prevents unstable summaries like:
        #   score=0.85 (safe) AND insufficient_ratio≈0.63 (review_required)
        contradiction_fatal = self.calibration.thresholds.get("contradiction_fatal", True)
        insufficient_threshold = self.calibration.thresholds.get("insufficient_evidence_ratio", 0.30)

        if contradiction_fatal and risk_dist["contradiction"] > 0:
            contract_score = min(contract_score, 0.39)
        elif insufficient_ratio > insufficient_threshold:
            # cap below the "low risk" band
            contract_score = min(contract_score, 0.64)

        legal_confidence = round(
            0.6 * grounding +
            0.4 * clarity,
            2
        )

        summary = ContractSummary(
            overall_score=contract_score,
            risk_level=self._risk_grade(contract_score),
            legal_confidence=legal_confidence,
            summary=self._summary_text(risk_dist, raw_dist, contract_score),
            distribution=ContractRiskDistribution(**risk_dist)
        )

        return ContractAnalysisResult(
            contract_summary=summary,
            top_issues=issues[:10],
            clauses=clauses
        )

    # =========================================================
    # SCORING COMPONENTS
    # =========================================================

    def _legal_core_score(self, clauses):
        if not clauses:
            return 1.0

        contradiction_ratio = sum(
            1 for c in clauses if c.alignment == "contradiction"
        ) / len(clauses)

        high_risk_ratio = sum(
            1 for c in clauses if c.risk_level == "high"
        ) / len(clauses)

        score = 1.0
        score -= 0.6 * contradiction_ratio
        score -= 0.3 * high_risk_ratio

        return max(0.0, score)

    def _clarity_score(self, dist, total):
        insufficient_ratio = dist["insufficient_evidence"] / total
        partial_ratio = dist["partially_aligned"] / total

        return max(0.0, 1.0 - 0.5 * insufficient_ratio - 0.3 * partial_ratio)

    def _avg_grounding(self, clauses):
        if not clauses:
            return 0.7
        scores = [
            (
                float(getattr(c, "groundedness_score"))
                if getattr(c, "groundedness_score", None) is not None
                else 0.7
            )
            for c in clauses
        ]
        return sum(scores) / len(scores)

    def _risk_grade(self, score: float) -> str:
        if score >= 0.75:
            return "low"
        if score >= 0.5:
            return "medium"
        return "high"

    # =========================================================
    # SUMMARY TEXT
    # =========================================================

    def _summary_text(self, risk_dist: dict, raw_dist: dict, score: float) -> str:
        return (
            f"The agreement was reviewed across {sum(raw_dist.values())} clauses. "
            f"{sum(risk_dist.values())} clauses materially affect legal rights. "
            f"{risk_dist['contradiction']} enforceable clauses present statutory conflicts. "
            f"The overall legal risk score of {score} reflects legal exposure, "
            f"drafting clarity, and statutory grounding."
        )

    # =========================================================
    # Issues extraction (used by UI + lawyer summary)
    # =========================================================

    def _build_top_issues(self, clauses: List[ClauseAnalysisResult]) -> List[KeyIssue]:
        """
        Build KeyIssue list from clause outputs.

        This is intentionally conservative and does not affect scoring.
        """
        out: List[KeyIssue] = []

        for c in clauses:
            # focus on enforceable clauses that are weak/unclear
            is_problem = (
                c.alignment in {"contradiction", "insufficient_evidence"}
                or float(c.quality_score) < 0.5
            )
            if not is_problem:
                continue

            statutory_anchor = self._statutory_anchor(c)
            evidence_reference = self._evidence_reference(c)
            evidence_snippet = (c.evidence_snippets[0] if getattr(c, "evidence_snippets", None) else None)

            issue_text = (
                c.issue_reason
                or self._default_issue_reason(
                    alignment=c.alignment,
                    statutory_anchor=statutory_anchor,
                    evidence_reference=evidence_reference,
                )
            )

            out.append(
                KeyIssue(
                    clause_id=c.clause_id,
                    display_reference=c.normalized_reference or f"Clause {c.clause_id}",
                    heading=c.heading,
                    risk_level=c.risk_level,
                    issue=issue_text,
                    statutory_anchor=statutory_anchor,
                    evidence_reference=evidence_reference,
                    evidence_snippet=evidence_snippet,
                    recommended_action=c.recommended_action or "Independent legal review is advised",
                    quality_score=float(round(c.quality_score, 2)),
                )
            )

        # sort worst first
        out.sort(key=lambda x: x.quality_score)
        return out

    def _statutory_anchor(self, clause: ClauseAnalysisResult) -> Optional[str]:
        refs = getattr(clause, "statutory_refs", None) or []
        if refs:
            return refs[0]
        # fall back to citations (prefer statute-like sources)
        for cit in getattr(clause, "citations", []) or []:
            source = str(cit.get("source", ""))
            ref = str(cit.get("ref", ""))
            if "rera" in source.lower():
                return f"{source} - {ref}" if ref else source
        return None

    def _evidence_reference(self, clause: ClauseAnalysisResult) -> Optional[str]:
        for cit in getattr(clause, "citations", []) or []:
            source = str(cit.get("source", ""))
            ref = str(cit.get("ref", ""))
            if "rera" not in source.lower() and ref:
                return f"{source} - {ref}"
        return None

    def _default_issue_reason(
        self,
        *,
        alignment: str,
        statutory_anchor: Optional[str],
        evidence_reference: Optional[str],
    ) -> str:
        if alignment == "contradiction":
            base = "Clause conflicts with statutory RERA protections"
        elif alignment == "insufficient_evidence":
            base = "Clause lacks clear statutory support or explicit rights"
        else:
            base = "Clause requires clarification to avoid legal ambiguity"

        details: List[str] = []
        if statutory_anchor:
            details.append(f"Anchor: {statutory_anchor}")
        if evidence_reference:
            details.append(f"Evidence: {evidence_reference}")
        return f"{base} ({'; '.join(details)})" if details else base