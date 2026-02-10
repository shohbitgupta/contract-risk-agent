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

from configs.callibration.callibration_config_loader import CalibrationConfig


# Clause roles that legally affect enforceability
RISK_RELEVANT_ROLES = {
    "obligation",
    "right",
    "procedure",
}


class ContractAggregationAgent:
    """
    Aggregates clause-level legal analysis into a contract-level,
    lawyer-defensible risk assessment.

    Core principles:
    - Only legally operative clauses affect risk
    - Definitions / schedules inform context, not risk
    - Worst-case exposure is prioritized
    """

    def __init__(self, calibration: CalibrationConfig):
        self.calibration = calibration

    # =========================================================
    # Public API
    # =========================================================

    def aggregate(
        self,
        clauses: List[ClauseAnalysisResult],
    ) -> ContractAnalysisResult:

        if not clauses:
            raise ValueError("Cannot aggregate empty clause list")

        ALIGNMENT_WEIGHTS = self.calibration.weights["alignment"]
        RISK_MULTIPLIERS = self.calibration.weights["risk_multiplier"]

        # -------------------------------------------------
        # Distributions
        # -------------------------------------------------
        raw_dist = {
            "aligned": 0,
            "partially_aligned": 0,
            "insufficient_evidence": 0,
            "contradiction": 0
        }

        risk_dist = raw_dist.copy()

        weighted_scores: List[float] = []
        issues: List[KeyIssue] = []

        # -------------------------------------------------
        # Clause-level evaluation
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

            clause_role = getattr(c, "clause_role", "unknown")

            # 🚫 Skip non-risk-bearing clauses
            if clause_role not in RISK_RELEVANT_ROLES:
                continue

            risk_dist[alignment] += 1

            risk = c.risk_level
            alignment_weight = ALIGNMENT_WEIGHTS.get(alignment, 0.4)
            risk_multiplier = RISK_MULTIPLIERS.get(risk, 0.85)

            semantic_conf = getattr(c, "semantic_confidence", 1.0)
            semantic_conf = max(0.0, min(1.0, semantic_conf))

            clause_score = round(
                c.quality_score
                * alignment_weight
                * risk_multiplier
                * (0.7 + 0.3 * semantic_conf),
                3
            )

            weighted_scores.append(clause_score)

            # -------------------------------------------------
            # Key issues (lawyer-facing)
            # -------------------------------------------------
            if clause_score < 0.5:
                issues.append(
                    KeyIssue(
                        clause_id=c.clause_id,
                        risk_level=c.risk_level,
                        issue=self._issue_reason(
                            alignment=alignment,
                            clause_role=clause_role
                        ),
                        recommended_action=(
                            c.recommended_action
                            or "Independent legal review is advised"
                        ),
                        quality_score=round(clause_score, 2)
                    )
                )

        # -------------------------------------------------
        # Schema guards
        # -------------------------------------------------
        assert set(raw_dist.keys()) == set(risk_dist.keys())
        assert sum(risk_dist.values()) <= sum(raw_dist.values())

        # -------------------------------------------------
        # Contract score (risk-bearing clauses only)
        # -------------------------------------------------
        if weighted_scores:
            contract_score = max(
                self._percentile_contract_score(weighted_scores),
                0.15
            )
        else:
            contract_score = 0.5  # Neutral if no enforceable clauses found

        risk_grade = self._risk_grade(contract_score)

        total_risk_clauses = sum(risk_dist.values()) or 1

        summary = ContractSummary(
            overall_score=contract_score,
            risk_level=risk_grade,
            legal_confidence=self._legal_confidence(
                contract_score,
                risk_dist,
                total_risk_clauses
            ),
            summary=self._summary_text(risk_dist, raw_dist, contract_score),
            distribution=ContractRiskDistribution(**risk_dist)
        )

        return ContractAnalysisResult(
            contract_summary=summary,
            top_issues=sorted(
                issues,
                key=lambda x: x.quality_score
            )[:10],
            clauses=clauses
        )

    # =========================================================
    # Scoring helpers
    # =========================================================

    def _percentile_contract_score(self, scores: List[float]) -> float:
        scores = sorted(scores)

        def percentile(p):
            k = max(0, min(len(scores) - 1, math.floor(p / 100 * len(scores))))
            return scores[k]

        return round(
            0.6 * percentile(20) + 0.4 * percentile(50),
            2
        )

    def _risk_grade(self, score: float) -> str:
        if score >= 0.8:
            return "low"
        if score >= 0.65:
            return "medium"
        return "high"

    def _legal_confidence(self, score: float, dist: dict, total: int) -> float:
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

    # =========================================================
    # Explanation helpers
    # =========================================================

    def _issue_reason(self, alignment: str, clause_role: str) -> str:
        role_prefix = {
            "obligation": "Promoter obligation",
            "right": "Allottee right",
            "procedure": "Contractual procedure",
        }.get(clause_role, "Clause")

        if alignment == "contradiction":
            return f"{role_prefix} conflicts with statutory RERA protections"

        if alignment == "insufficient_evidence":
            return f"{role_prefix} lacks clear statutory support or explicit rights"

        return f"{role_prefix} requires clarification to avoid legal ambiguity"

    def _summary_text(self, risk_dist: dict, raw_dist: dict, score: float) -> str:
        return (
            f"The agreement was reviewed across {sum(raw_dist.values())} clauses, "
            f"of which {sum(risk_dist.values())} clauses materially affect legal rights "
            f"and obligations. "
            f"{risk_dist['contradiction']} enforceable clauses present potential "
            f"statutory conflicts, and {risk_dist['insufficient_evidence']} enforceable "
            f"clauses lack sufficient clarity under RERA. "
            f"The overall legal risk score of {score} reflects a conservative, "
            f"worst-case assessment of enforceability."
        )
