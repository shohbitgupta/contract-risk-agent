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
        groundedness_values: List[float] = []
        semantic_values: List[float] = []

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

            groundedness = getattr(c, "groundedness_score", None)
            groundedness = (
                float(groundedness)
                if groundedness is not None
                else 0.5
            )
            groundedness = max(0.0, min(1.0, groundedness))

            groundedness_values.append(groundedness)
            semantic_values.append(semantic_conf)

            # -------------------------------------------------
            # Clause score design (key change)
            # -------------------------------------------------
            # Your earlier scoring made the contract score heavily dependent on
            # `quality_score`, which in turn is often capped by compliance confidence.
            # When grounding is strong but the model is cautious, scores stayed near ~0.
            #
            # New approach:
            # - `quality_score` still matters (interpretation confidence)
            # - `groundedness_score` lifts score when evidence is clearly anchored
            # - `semantic_confidence` contributes modestly (chunk quality)
            #
            # Result: evidence-backed clauses no longer get "stuck" at ~0.05–0.10.
            combined_conf = (
                0.45 * float(c.quality_score)
                + 0.35 * groundedness
                + 0.20 * semantic_conf
            )
            combined_conf = max(0.0, min(1.0, combined_conf))

            clause_score = round(
                combined_conf * alignment_weight * risk_multiplier,
                3,
            )


            weighted_scores.append(clause_score)

            # -------------------------------------------------
            # Key issues (lawyer-facing)
            # -------------------------------------------------
            if clause_score < 0.5:
                statutory_anchor = self._extract_statutory_anchor(c)
                evidence_reference = self._extract_evidence_reference(c)
                evidence_snippet = self._extract_evidence_snippet(c)
                issues.append(
                    KeyIssue(
                        clause_id=c.clause_id,
                        display_reference=(
                            c.normalized_reference or f"Clause {c.clause_id}"
                        ),
                        heading=c.heading,
                        risk_level=c.risk_level,
                        issue=self._issue_reason(
                            alignment=alignment,
                            clause_role=clause_role,
                            statutory_anchor=statutory_anchor,
                            evidence_reference=evidence_reference,
                        ),
                        statutory_anchor=statutory_anchor,
                        evidence_reference=evidence_reference,
                        evidence_snippet=evidence_snippet,
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
            contract_score = self._percentile_contract_score(weighted_scores)
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
                total_risk_clauses,
                avg_groundedness=(
                    round(sum(groundedness_values) / len(groundedness_values), 2)
                    if groundedness_values else 0.0
                ),
                avg_semantic_conf=(
                    round(sum(semantic_values) / len(semantic_values), 2)
                    if semantic_values else 0.0
                ),
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

    def _legal_confidence(
        self,
        score: float,
        dist: dict,
        total: int,
        *,
        avg_groundedness: float,
        avg_semantic_conf: float,
    ) -> float:
        if total <= 0:
            return 0.0

        aligned_ratio = dist["aligned"] / total
        contradiction_ratio = dist["contradiction"] / total
        unclear_ratio = dist["insufficient_evidence"] / total

        # Confidence reflects reliability/grounding, not just risk level.
        # A risky contract can still have high confidence if the evidence is clear.
        confidence = (
            0.45 * max(0.0, min(1.0, avg_groundedness))
            + 0.20 * max(0.0, min(1.0, avg_semantic_conf))
            + 0.15 * aligned_ratio
            + 0.10 * (1.0 - unclear_ratio)
            + 0.10 * (1.0 - contradiction_ratio)
        )

        return round(max(0.0, min(1.0, confidence)), 2)

    # =========================================================
    # Explanation helpers
    # =========================================================

    def _issue_reason(
        self,
        alignment: str,
        clause_role: str,
        statutory_anchor: str | None = None,
        evidence_reference: str | None = None,
    ) -> str:
        role_prefix = {
            "obligation": "Promoter obligation",
            "right": "Allottee right",
            "procedure": "Contractual procedure",
        }.get(clause_role, "Clause")

        if alignment == "contradiction":
            base = f"{role_prefix} conflicts with statutory RERA protections"
        elif alignment == "insufficient_evidence":
            base = f"{role_prefix} lacks clear statutory support or explicit rights"
        else:
            base = f"{role_prefix} requires clarification to avoid legal ambiguity"

        details = []
        if statutory_anchor:
            details.append(f"Anchor: {statutory_anchor}")
        if evidence_reference:
            details.append(f"Evidence: {evidence_reference}")

        if details:
            return f"{base} ({'; '.join(details)})"
        return base

    def _extract_statutory_anchor(self, clause: ClauseAnalysisResult) -> str | None:
        if getattr(clause, "statutory_refs", None):
            return clause.statutory_refs[0]

        for citation in clause.citations:
            source = str(citation.get("source", ""))
            ref = str(citation.get("ref", ""))
            if "rera" in source.lower():
                return f"{source} - {ref}" if ref else source
        return None

    def _extract_evidence_reference(self, clause: ClauseAnalysisResult) -> str | None:
        for citation in clause.citations:
            source = str(citation.get("source", ""))
            ref = str(citation.get("ref", ""))
            if "rera" not in source.lower() and ref:
                return f"{source} - {ref}"
        return None

    def _extract_evidence_snippet(self, clause: ClauseAnalysisResult) -> str | None:
        snippets = getattr(clause, "evidence_snippets", []) or []
        if snippets:
            return snippets[0]
        return None

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
