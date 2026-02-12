from typing import List
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
            getattr(c, "groundedness", 0.7)
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