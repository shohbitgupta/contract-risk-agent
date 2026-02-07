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
    # Verdict logic (calibrated)
    # -----------------------------
    if (
        calibration.thresholds.get("contradiction_fatal", False)
        and dist.contradiction > 0
    ):
        verdict = "do_not_sign"
        headline = "High legal risk: agreement should not be relied upon as drafted."

    elif (
        total > 0
        and dist.insufficient_evidence
        > total * calibration.thresholds.get("insufficient_evidence_ratio", 0.25)
    ):
        verdict = "review_required"
        headline = "Moderate legal risk: key statutory protections are unclear."

    else:
        verdict = "safe_to_sign"
        headline = "Low legal risk: agreement broadly aligns with RERA protections."

    # -----------------------------
    # Why this matters (legal reasoning)
    # -----------------------------
    why = []

    if dist.contradiction > 0:
        why.append(
            f"{dist.contradiction} clauses directly conflict with statutory RERA protections."
        )

    if dist.insufficient_evidence > 0:
        why.append(
            f"{dist.insufficient_evidence} clauses do not clearly articulate mandatory statutory rights."
        )

    if dist.partially_aligned > 0:
        why.append(
            f"{dist.partially_aligned} clauses rely on implicit or partial compliance, increasing interpretation risk."
        )

    # -----------------------------
    # Key risk statistics
    # -----------------------------
    stats = [
        f"Clauses reviewed: {total}",
        f"Overall legal risk score: {score} (scale: 0 = high risk, 1 = low risk)",
        f"High-risk clauses (score < 0.5): {len([i for i in issues if i.quality_score < 0.5])}",
    ]

    # -----------------------------
    # Critical clauses (top 5)
    # -----------------------------
    critical = [
        (
            f"Clause {i.clause_id}: {i.issue} "
            f"(risk: {i.risk_level}, score: {i.quality_score})"
        )
        for i in issues[:5]
    ]

    # -----------------------------
    # Recommended actions
    # -----------------------------
    actions = []

    if verdict == "do_not_sign":
        actions.extend([
            "Do not sign this agreement in its current form.",
            "Renegotiate clauses that conflict with statutory RERA protections.",
            "Seek formal legal advice before execution."
        ])
    elif verdict == "review_required":
        actions.extend([
            "Request clarification or redrafting of ambiguous clauses.",
            "Ensure statutory RERA rights are explicitly stated.",
            "Review identified high-risk clauses before signing."
        ])
    else:
        actions.append(
            "Agreement appears legally acceptable; retain documentation for records."
        )

    return LawyerFriendlySummary(
        verdict=verdict,
        headline=headline,
        why_this_matters=why,
        key_risk_statistics=stats,
        critical_clauses=critical,
        recommended_next_steps=actions
    )