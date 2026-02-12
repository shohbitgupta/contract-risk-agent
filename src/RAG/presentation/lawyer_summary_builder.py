from typing import List, Any

from RAG.contract_analysis import ContractAnalysisResult
from RAG.presentation.lawyer_summary import LawyerFriendlySummary


# Clause roles that materially affect enforceability
RISK_RELEVANT_ROLES = {
    "obligation",
    "right",
    "procedure",
}


def build_lawyer_friendly_summary(
    analysis: ContractAnalysisResult,
    calibration: Any = None,
) -> LawyerFriendlySummary:
    """
    Converts a ContractAnalysisResult into a lawyer-grade,
    opinion-style summary.

    Principles:
    - Contradictions are fatal
    - Ambiguity is assessed proportionally
    - Score is secondary signal
    - No ML/statistical language
    """

    summary = analysis.contract_summary
    dist = summary.distribution
    score = summary.overall_score
    issues = analysis.top_issues

    # -------------------------------------------------
    # Determine enforceable clause universe
    # -------------------------------------------------
    enforceable_clauses = [
        c for c in analysis.clauses
        if getattr(c, "clause_role", None) in RISK_RELEVANT_ROLES
    ]

    total_enforceable = len(enforceable_clauses)

    # -------------------------------------------------
    # Safety fallback
    # -------------------------------------------------
    if total_enforceable == 0:
        return LawyerFriendlySummary(
            verdict="review_required",
            headline="Contract requires legal review due to structural ambiguity.",
            why_this_matters=[
                "The agreement does not clearly identify enforceable obligations "
                "or rights suitable for legal risk evaluation."
            ],
            key_risk_statistics=[
                "Enforceable clauses identified: 0",
                "Risk assessment could not be reliably completed",
            ],
            critical_clauses=[],
            recommended_next_steps=[
                "Seek legal review to identify enforceable obligations.",
                "Clarify contract structure and clause numbering.",
            ],
        )

    # -------------------------------------------------
    # Calibration thresholds
    # -------------------------------------------------
    contradiction_fatal = True
    insufficient_ratio_threshold = 0.30  # default

    if calibration:
        contradiction_fatal = calibration.thresholds.get(
            "contradiction_fatal",
            True
        )
        insufficient_ratio_threshold = calibration.thresholds.get(
            "insufficient_evidence_ratio",
            insufficient_ratio_threshold
        )

    # -------------------------------------------------
    # Derived ratios
    # -------------------------------------------------
    insufficient_ratio = (
        dist.insufficient_evidence / total_enforceable
        if total_enforceable > 0 else 0
    )

    partially_ratio = (
        dist.partially_aligned / total_enforceable
        if total_enforceable > 0 else 0
    )

    # -------------------------------------------------
    # Verdict logic (layered, lawyer-aligned)
    # -------------------------------------------------

    # 🔴 Fatal statutory contradiction
    if contradiction_fatal and dist.contradiction > 0:
        verdict = "do_not_sign"
        headline = (
            "High legal risk: one or more enforceable clauses "
            "conflict with mandatory RERA protections."
        )

    # 🟠 Material ambiguity (ratio-based)
    elif insufficient_ratio > insufficient_ratio_threshold:
        verdict = "review_required"
        headline = (
            "Moderate legal risk: multiple enforceable clauses "
            "lack clear statutory alignment."
        )

    # 🟠 Score-based fallback
    elif score < 0.5:
        verdict = "review_required"
        headline = (
            "Moderate legal risk: enforceability concerns require review."
        )

    # 🟢 Broad compliance
    else:
        verdict = "safe_to_sign"
        headline = (
            "Low legal risk: enforceable clauses broadly align "
            "with RERA requirements."
        )

    # -------------------------------------------------
    # Why this matters (legal reasoning)
    # -------------------------------------------------
    why: List[str] = []

    if dist.contradiction > 0:
        why.append(
            f"{dist.contradiction} enforceable clause(s) directly conflict "
            f"with mandatory provisions of the RERA Act."
        )

    if insufficient_ratio > 0:
        why.append(
            f"{dist.insufficient_evidence} enforceable clause(s) do not "
            f"explicitly preserve statutory rights, increasing litigation risk."
        )

    if partially_ratio > 0:
        why.append(
            f"{dist.partially_aligned} enforceable clause(s) rely on implicit "
            f"statutory incorporation rather than clear drafting."
        )

    if not why:
        why.append(
            "No material statutory conflicts were identified in enforceable clauses."
        )

    # -------------------------------------------------
    # Key risk statistics (lawyer-readable)
    # -------------------------------------------------
    stats = [
        f"Total clauses reviewed: {len(analysis.clauses)}",
        f"Enforceable clauses assessed: {total_enforceable}",
        f"Overall legal risk score: {score} (0 = high risk, 1 = low risk)",
        f"High-risk enforceable clauses: "
        f"{len([i for i in issues if i.quality_score < 0.5])}",
    ]

    # -------------------------------------------------
    # Critical clauses (top 5)
    # -------------------------------------------------
    critical = [
        f"{i.display_reference or ('Clause ' + i.clause_id)}"
        f"{(' - ' + i.heading) if i.heading else ''}: {i.issue} "
        f"(risk level: {i.risk_level}, score: {i.quality_score})"
        for i in issues[:5]
    ]

    # -------------------------------------------------
    # Recommended legal actions
    # -------------------------------------------------
    actions: List[str] = []

    if verdict == "do_not_sign":
        actions.extend([
            "Do not execute the agreement in its current form.",
            "Seek immediate legal advice on clauses conflicting with RERA.",
            "Require redrafting to explicitly preserve statutory rights.",
        ])

    elif verdict == "review_required":
        actions.extend([
            "Seek clarification or redrafting of ambiguous enforceable clauses.",
            "Ensure statutory rights under RERA are explicitly incorporated.",
            "Review high-risk clauses before execution.",
        ])

    else:
        actions.append(
            "No immediate legal action required; retain a copy for records."
        )

    # -------------------------------------------------
    # Final output
    # -------------------------------------------------
    return LawyerFriendlySummary(
        verdict=verdict,
        headline=headline,
        why_this_matters=why,
        key_risk_statistics=stats,
        critical_clauses=critical,
        recommended_next_steps=actions,
    )
