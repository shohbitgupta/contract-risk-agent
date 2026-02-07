# src/RAG/presentation/lawyer_summary_builder.py

from RAG.presentation.lawyer_summary import LawyerFriendlySummary
from RAG.contract_analysis import ContractAnalysisResult
from configs.callibration.callibration_config_loader import CalibrationConfig


def build_lawyer_friendly_summary(
    analysis: ContractAnalysisResult,
    calibration: CalibrationConfig
) -> LawyerFriendlySummary:

    dist = analysis.contract_summary.distribution
    score = analysis.contract_summary.overall_score
    issues = analysis.top_issues

    total = sum(dist.model_dump().values())

    # -----------------------------
    # Verdict logic
    # -----------------------------
    if calibration.thresholds.get("contradiction_fatal") and dist.contradiction > 0:
        verdict = "do_not_sign"
        headline = "High legal risk: agreement should not be relied upon as drafted."

    elif dist.insufficient_evidence > total * calibration.thresholds.get(
        "insufficient_evidence_ratio", 0.25
    ):
        verdict = "review_required"
        headline = "Moderate legal risk: key statutory protections are unclear."

    else:
        verdict = "safe_to_sign"
        headline = "Low legal risk: agreement broadly aligns with RERA protections."

    # -----------------------------
    # Why this matters
    # -----------------------------
    why = []

    if dist.contradiction:
        why.append(
            f"{dist.contradiction} clauses directly conflict with mandatory RERA provisions."
        )

    if dist.insufficient_evidence:
        why.append(
            f"{dist.insufficient_evidence} clauses fail to explicitly preserve statutory rights "
            f"(e.g. Sections 14 and 18 of the RERA Act)."
        )

    if dist.partially_aligned:
        why.append(
            f"{dist.partially_aligned} clauses rely on implicit compliance, increasing interpretation risk."
        )

    # -----------------------------
    # Key stats
    # -----------------------------
    stats = [
        f"Clauses reviewed: {total}",
        f"Overall legal risk score: {score} (0 = high risk, 1 = low risk)",
        f"High-risk clauses (score < 0.5): {len([i for i in issues if i.quality_score < 0.5])}",
    ]

    # -----------------------------
    # Critical clauses (lawyer-grade)
    # -----------------------------
    critical = []
    for i in issues[:5]:
        ref = getattr(i, "normalized_reference", None) or f"Clause {i.clause_id}"
        heading = getattr(i, "heading", None)
        label = f"{ref}"
        if heading:
            label += f" – {heading}"

        critical.append(
            f"{label}: {i.issue} (risk: {i.risk_level}, score: {i.quality_score})"
        )

    # -----------------------------
    # Recommended actions
    # -----------------------------
    actions = []

    if verdict == "do_not_sign":
        actions.extend([
            "Do not sign the agreement without renegotiation.",
            "Remove or amend clauses conflicting with RERA.",
            "Seek formal legal opinion."
        ])
    elif verdict == "review_required":
        actions.extend([
            "Seek clarification or redrafting of ambiguous clauses.",
            "Ensure statutory RERA rights are explicitly stated.",
            "Review high-risk clauses before execution."
        ])
    else:
        actions.append(
            "Agreement appears legally acceptable; retain executed copy for records."
        )

    return LawyerFriendlySummary(
        verdict=verdict,
        headline=headline,
        why_this_matters=why,
        key_risk_statistics=stats,
        critical_clauses=critical,
        recommended_next_steps=actions
    )
