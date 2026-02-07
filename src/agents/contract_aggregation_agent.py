from typing import List
import math
from constants.alignment import ALLOWED_ALIGNMENTS

from RAG.contract_analysis import (
    ContractAnalysisResult,
    ContractSummary,
    ContractRiskDistribution,
    KeyIssue,
    ClauseAnalysisResult
)


ALIGNMENT_WEIGHTS = {
    "aligned": 1.0,
    "partially_aligned": 0.7,
    "insufficient_evidence": 0.4,
    "contradiction": 0.0
}

RISK_MULTIPLIERS = {
    "low": 1.0,
    "medium": 0.85,
    "high": 0.6
}


class ContractAggregationAgent:
    """
    Aggregates clause-level legal analysis into a contract-level,
    lawyer-defensible risk assessment.
    """

    def aggregate(
        self,
        clauses: List[ClauseAnalysisResult]
    ) -> ContractAnalysisResult:

        if not clauses:
            raise ValueError("Cannot aggregate empty clause list")

        # -----------------------------
        # Distribution (contract-level)
        # -----------------------------
        dist = {
            "aligned": 0,
            "partially_aligned": 0,
            "insufficient_evidence": 0,
            "contradiction": 0
        }

        weighted_scores: List[float] = []
        issues: List[KeyIssue] = []

        total_clauses = len(clauses)

        # -----------------------------
        # Clause-level scoring
        # -----------------------------
        for c in clauses:
            alignment = c.alignment
            if alignment == "conflicting":
                alignment = "contradiction"

            if alignment not in ALLOWED_ALIGNMENTS:
                raise ValueError(
                    f"Invalid alignment '{alignment}' for clause {c.clause_id}"
                )

            risk = c.risk_level

            # ✅ single source of truth
            dist[alignment] += 1

            alignment_weight = ALIGNMENT_WEIGHTS.get(alignment, 0.4)
            risk_multiplier = RISK_MULTIPLIERS.get(risk, 0.85)

            clause_score = round(
                c.quality_score * alignment_weight * risk_multiplier,
                3
            )

            weighted_scores.append(clause_score)

            if clause_score < 0.5:
                issues.append(
                    KeyIssue(
                        clause_id=c.clause_id,
                        risk_level=c.risk_level,
                        issue=self._issue_reason(alignment),
                        recommended_action=(
                            c.recommended_action
                            or "Independent legal review is advised"
                        ),
                        quality_score=round(clause_score, 2)
                    )
                )

        # -----------------------------
        # Schema guard
        # -----------------------------
        assert set(dist.keys()) == {
            "aligned",
            "partially_aligned",
            "insufficient_evidence",
            "contradiction"
        }, f"Invalid distribution keys: {dist}"

        # -----------------------------
        # Contract score
        # -----------------------------
        contract_score = self._percentile_contract_score(weighted_scores)
        risk_grade = self._risk_grade(contract_score)

        summary = ContractSummary(
            overall_score=contract_score,
            risk_level=risk_grade,
            legal_confidence=self._legal_confidence(contract_score, dist, total_clauses),
            summary=self._summary_text(dist, contract_score),
            distribution=ContractRiskDistribution(**dist)
        )

        return ContractAnalysisResult(
            contract_summary=summary,
            top_issues=sorted(
                issues,
                key=lambda x: x.quality_score
            )[:10],
            clauses=clauses
        )

    # -------------------------------------------------
    # Scoring helpers
    # -------------------------------------------------

    def _percentile_contract_score(self, scores: List[float]) -> float:
        scores = sorted(scores)

        def percentile(p):
            if not scores:
                return 0.0
            k = max(0, min(len(scores) - 1, math.floor(p / 100 * len(scores))))
            return scores[k]

        return round(0.6 * percentile(20) + 0.4 * percentile(50), 2)

    def _risk_grade(self, score: float) -> str:
        if score >= 0.8:
            return "low"
        if score >= 0.65:
            return "medium"
        return "high"

    def _legal_confidence(self, score: float, dist: dict, total: int) -> float:
        """
        Scale-safe legal confidence.
        Penalizes ambiguity proportionally, not absolutely.
        """
        aligned_ratio = dist["aligned"] / total
        contradiction_ratio = dist["contradiction"] / total
        unclear_ratio = dist["insufficient_evidence"] / total

        confidence = (
            score
            + 0.15 * aligned_ratio
            - 0.25 * contradiction_ratio
            - 0.15 * unclear_ratio
        )

        return round(max(0.0, min(1.0, confidence)), 2)

    # -------------------------------------------------
    # Explanation helpers
    # -------------------------------------------------

    def _issue_reason(self, alignment: str) -> str:
        if alignment == "contradiction":
            return "Clause conflicts with statutory RERA protections"
        if alignment == "insufficient_evidence":
            return "Clause lacks clear statutory support or explicit rights"
        return "Clause requires clarification to avoid legal ambiguity"

    def _summary_text(self, dist: dict, score: float) -> str:
        return (
            f"The agreement was evaluated across {sum(dist.values())} clauses. "
            f"{dist['contradiction']} clauses present potential statutory conflicts, "
            f"and {dist['insufficient_evidence']} clauses lack sufficient clarity "
            f"under RERA. The overall legal risk score of {score} reflects a "
            f"conservative assessment prioritizing worst-case legal exposure."
        )
