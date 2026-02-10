import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import main as run_contract_analysis  # noqa: E402


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
                    st.write(f"- **Statutory anchor:** {issue['statutory_anchor']}")
                if issue.get("evidence_reference"):
                    st.write(f"- **Retrieved reference:** {issue['evidence_reference']}")
        else:
            st.write("No top issues identified.")

        output_json = json.dumps(result, indent=2)
        st.download_button(
            "Download JSON Report",
            data=output_json,
            file_name=f"contract_analysis_{selected_state}.json",
            mime="application/json",
        )

        st.subheader("Full JSON Output")
        st.code(output_json, language="json")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
