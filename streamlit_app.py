import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import re


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import main as run_contract_analysis  # noqa: E402


_SECTION_CITE_RE = re.compile(r"(Section\s+\d+[A-Za-z]*\b(?:\(\d+\))?)", re.IGNORECASE)


def _highlight_citations(text: str) -> str:
    """
    Render-friendly citation highlighting for statute anchors like:
    - Section 18
    - Section 18(1)
    - Section 18A
    """
    if not text:
        return ""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return _SECTION_CITE_RE.sub(r"<mark>\1</mark>", safe)


def _available_states() -> list[str]:
    vector_state_dir = SRC_DIR / "data" / "vector_indexes"
    if not vector_state_dir.exists():
        return ["uttar_pradesh"]
    return sorted(
        [p.name for p in vector_state_dir.iterdir() if p.is_dir()]
    ) or ["uttar_pradesh"]


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.get("contract_summary", {})
    clauses = result.get("clauses", [])
    top_issues = result.get("top_issues", [])
    distribution = summary.get("distribution", {})

    return {
        "overall_score": summary.get("overall_score"),
        "risk_level": summary.get("risk_level"),
        "legal_confidence": summary.get("legal_confidence"),
        "clauses_analyzed": len(clauses),
        "top_issues_count": len(top_issues),
        "distribution": distribution,
    }


def _grounding_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    clauses = result.get("clauses", [])
    if not clauses:
        return {
            "avg_groundedness": None,
            "low_grounded_count": 0,
            "insufficient_evidence_count": 0,
            "contradiction_count": 0,
            "low_grounded_clauses": [],
        }

    groundedness_values = [
        c.get("groundedness_score")
        for c in clauses
        if c.get("groundedness_score") is not None
    ]

    low_grounded = [
        c for c in clauses
        if c.get("groundedness_score") is not None and c.get("groundedness_score", 1.0) < 0.4
    ]

    return {
        "avg_groundedness": (
            round(sum(groundedness_values) / len(groundedness_values), 2)
            if groundedness_values else None
        ),
        "low_grounded_count": len(low_grounded),
        "insufficient_evidence_count": sum(
            1 for c in clauses if c.get("alignment") == "insufficient_evidence"
        ),
        "contradiction_count": sum(
            1 for c in clauses if c.get("alignment") == "contradiction"
        ),
        "low_grounded_clauses": low_grounded[:5],
    }


def main() -> None:
    st.set_page_config(
        page_title="Contract Risk Analyzer",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("Contract Risk Analyzer")
    st.caption("Upload a contract PDF and run grounded legal-risk analysis.")

    states = _available_states()
    selected_state = st.selectbox(
        "Select jurisdiction/state",
        options=states,
        index=states.index("uttar_pradesh") if "uttar_pradesh" in states else 0,
    )

    uploaded_file = st.file_uploader(
        "Upload contract PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

    analyze_clicked = st.button("Start Analyzing", type="primary")

    if analyze_clicked and not uploaded_file:
        st.warning("Please upload a PDF before starting analysis.")
        return

    if not analyze_clicked:
        st.info("Upload a contract PDF and click 'Start Analyzing'.")
        return

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    try:
        with st.spinner("Analyzing contract... this may take a minute."):
            result = run_contract_analysis(str(temp_path), state=selected_state)

        st.success("Analysis completed.")

        summary = _summarize_result(result)
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", summary["overall_score"])
        col2.metric("Risk Level", summary["risk_level"])
        col3.metric("Legal Confidence", summary["legal_confidence"])

        st.subheader("Distribution")
        st.json(summary["distribution"])

        st.subheader("Grounding Diagnostics")
        grounding = _grounding_diagnostics(result)
        gd1, gd2, gd3, gd4 = st.columns(4)
        gd1.metric("Avg Groundedness", grounding["avg_groundedness"] if grounding["avg_groundedness"] is not None else "N/A")
        gd2.metric("Low Grounded Clauses", grounding["low_grounded_count"])
        gd3.metric("Insufficient Evidence", grounding["insufficient_evidence_count"])
        gd4.metric("Contradictions", grounding["contradiction_count"])

        if grounding["low_grounded_clauses"]:
            with st.expander("Why confidence is low (sample clauses)"):
                for clause in grounding["low_grounded_clauses"]:
                    ref = clause.get("normalized_reference") or f"Clause {clause.get('clause_id', 'N/A')}"
                    heading = clause.get("heading")
                    if heading:
                        ref = f"{ref} - {heading}"
                    st.markdown(f"**{ref}**")
                    st.caption(
                        f"Alignment: {clause.get('alignment', 'N/A')} | "
                        f"Groundedness: {clause.get('groundedness_score', 'N/A')} | "
                        f"Quality: {clause.get('quality_score', 'N/A')}"
                    )
                    if clause.get("issue_reason"):
                        st.write(f"- {clause['issue_reason']}")

        st.subheader("Lawyer Summary")
        lawyer_summary = result.get("lawyer_summary", {})
        if lawyer_summary:
            st.markdown(f"**Verdict:** {lawyer_summary.get('verdict', 'N/A')}")
            st.markdown(f"**Headline:** {lawyer_summary.get('headline', 'N/A')}")

            if lawyer_summary.get("why_this_matters"):
                st.markdown("**Why this matters**")
                for item in lawyer_summary["why_this_matters"]:
                    st.write(f"- {item}")

            if lawyer_summary.get("recommended_next_steps"):
                st.markdown("**Recommended next steps**")
                for step in lawyer_summary["recommended_next_steps"]:
                    st.write(f"- {step}")
        else:
            st.caption("Lawyer summary not available in response.")

        st.subheader("Top Issues")
        top_issues = result.get("top_issues", [])
        if top_issues:
            seen = set()
            shown = 0
            for issue in top_issues:
                dedup_key = (
                    issue.get("display_reference") or issue.get("clause_id"),
                    issue.get("issue"),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                shown += 1
                if shown > 10:
                    break

                ref = issue.get("display_reference") or f"Clause {issue.get('clause_id', 'N/A')}"
                heading = issue.get("heading")
                if heading:
                    ref = f"{ref} - {heading}"

                st.markdown(
                    f"**{shown}. {ref}** - "
                    f"{issue.get('issue', 'Issue not available')}"
                )
                st.caption(
                    f"Risk: {issue.get('risk_level', 'N/A')} | "
                    f"Score: {issue.get('quality_score', 'N/A')}"
                )
                if issue.get("statutory_anchor"):
                    st.markdown(
                        f"- **Statutory anchor:** {_highlight_citations(str(issue['statutory_anchor']))}",
                        unsafe_allow_html=True,
                    )
                if issue.get("evidence_reference"):
                    st.markdown(
                        f"- **Retrieved reference:** {_highlight_citations(str(issue['evidence_reference']))}",
                        unsafe_allow_html=True,
                    )
                if issue.get("evidence_snippet"):
                    st.markdown(
                        f"- **RERA segment:** {_highlight_citations(str(issue['evidence_snippet']))}",
                        unsafe_allow_html=True,
                    )
        else:
            st.write("No top issues identified.")

        output_json = json.dumps(result, indent=2)
        st.download_button(
            "Download JSON Report",
            data=output_json,
            file_name=f"contract_analysis_{selected_state}.json",
            mime="application/json",
        )

        # Keep UI concise: downloadable report only.

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
