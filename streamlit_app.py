import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import re
from datetime import datetime
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import main as run_contract_analysis  # noqa: E402
from vector_index.embedding import EmbeddingGenerator  # noqa: E402
from vector_index.index_registry import IndexRegistry  # noqa: E402
from retrieval.reranking_agent import CrossEncoderReRankingAgent  # noqa: E402
from agents.legal_chat_agent import (  # noqa: E402
    OllamaLegalChatAgent,
    ChatClauseContext,
    ChatSourceContext,
)


_SECTION_CITE_RE = re.compile(r"(Section\s+\d+[A-Za-z]*\b(?:\(\d+\))?)", re.IGNORECASE)
_RULE_CITE_RE = re.compile(r"(Rule\s+\d+\b(?:\(\d+\))?)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+[a-zA-Z]*")

_CUSTOM_CSS = """
<style>
  /* Layout */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }

  /* Typography */
  h1, h2, h3 { letter-spacing: -0.02em; }

  /* Mark highlight (citations) */
  mark {
    background: rgba(255, 224, 130, 0.7);
    padding: 0.05rem 0.25rem;
    border-radius: 0.35rem;
  }

  /* Card style sections */
  .cr-card {
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 0.9rem;
    padding: 1rem 1rem;
    background: rgba(255, 255, 255, 0.02);
  }
  .cr-muted { color: rgba(240,240,240,0.72); }
  .cr-small { font-size: 0.92rem; }
</style>
"""


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
    safe = _SECTION_CITE_RE.sub(r"<mark>\1</mark>", safe)
    safe = _RULE_CITE_RE.sub(r"<mark>\1</mark>", safe)
    return safe


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))

def _format_citations(citations: list[dict[str, Any]] | None) -> str:
    """
    Render citations with statute-first ordering.
    Expected item shape: { "source": str, "ref": str }
    """
    if not citations:
        return ""

    statutory: list[str] = []
    other: list[str] = []
    for c in citations:
        source = str(c.get("source", "")).strip()
        ref = str(c.get("ref", "")).strip()
        if not source and not ref:
            continue
        item = f"{source} - {ref}" if ref else source
        if "rera" in source.lower() or "rera" in ref.lower():
            statutory.append(item)
        else:
            other.append(item)

    ordered = statutory + other
    ordered = ordered[:10]
    bullets = "\n".join(f"- {_highlight_citations(x)}" for x in ordered if x)
    return bullets


def _clause_blob(clause: Dict[str, Any]) -> str:
    """
    Build a searchable text blob for report-aware Q&A.
    """
    parts = [
        clause.get("normalized_reference") or clause.get("clause_id") or "",
        clause.get("heading") or "",
        clause.get("plain_summary") or "",
        clause.get("legal_explanation") or "",
        " ".join(clause.get("statutory_refs") or []),
        " ".join(clause.get("evidence_snippets") or []),
    ]
    return "\n".join(p for p in parts if p)


def _top_relevant_clauses(
    question: str,
    clauses: list[Dict[str, Any]],
    *,
    k: int = 5,
) -> list[Dict[str, Any]]:
    """
    Lightweight lexical relevance ranking over analyzed clause outputs.
    Designed to stay grounded by only using the report itself.
    """
    q = (question or "").strip()
    if not q or not clauses:
        return []

    q_tokens = _tokens(q)
    scored: list[tuple[float, Dict[str, Any]]] = []

    for c in clauses:
        blob = _clause_blob(c)
        c_tokens = _tokens(blob)
        if not c_tokens:
            continue

        overlap = len(q_tokens.intersection(c_tokens))
        # Extra bump for explicit statute/rule mentions
        statute_bonus = 0.0
        if "section" in q.lower() and _SECTION_CITE_RE.search(blob):
            statute_bonus += 1.0
        if "rule" in q.lower() and _RULE_CITE_RE.search(blob):
            statute_bonus += 1.0

        # Extra bump for exact substring hit
        exact_bonus = 2.0 if q.lower() in blob.lower() else 0.0

        # Prefer clauses that are enforceable / problematic
        align = (c.get("alignment") or "").lower()
        align_bonus = 0.5 if align in {"insufficient_evidence", "contradiction"} else 0.0

        score = overlap + statute_bonus + exact_bonus + align_bonus
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]

@st.cache_resource(show_spinner=False)
def _qa_index_registry() -> IndexRegistry:
    # Matches `src/main.py` runtime expectation
    base_dir = SRC_DIR / "data" / "vector_indexes"
    return IndexRegistry(base_dir=base_dir, embedding_dim=384)


@st.cache_resource(show_spinner=False)
def _qa_embedder() -> EmbeddingGenerator:
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def _qa_reranker() -> CrossEncoderReRankingAgent:
    return CrossEncoderReRankingAgent(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=8)


def _semantic_retrieve(
    question: str,
    *,
    state: str,
    top_k: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Semantic retrieval over the same FAISS indexes used by the system.
    Returns a list of dicts with {source, ref, doc_type, snippet, chunk_id}.
    """
    q = (question or "").strip()
    if not q:
        return [], {"reason": "empty_question"}

    debug: dict[str, Any] = {"state": state, "question": q, "top_k": top_k}

    registry = _qa_index_registry()
    try:
        indexes = registry.get_indexes(state)
    except Exception as e:
        debug["error"] = f"failed_to_load_indexes: {e}"
        print(f"[chat-retrieval] failed to load indexes for state={state}: {e}")
        return [], debug
    debug["indexes_loaded"] = sorted(list(indexes.keys()))

    embedder = _qa_embedder()
    q_emb = np.asarray(embedder.embed([q])[0], dtype="float32")

    # Search all available indexes; prefer statute ones first.
    priority = ["rera_act", "rera_rules", "case_law", "circulars", "model_bba"]
    index_names = [n for n in priority if n in indexes] + [n for n in indexes.keys() if n not in priority]

    candidates = []
    per_index_counts: dict[str, int] = {}
    for name in index_names:
        idx = indexes[name]
        docs = idx.search(query_embedding=q_emb, top_k=18)
        candidates.extend(docs)
        per_index_counts[name] = len(docs)
    debug["per_index_hits"] = per_index_counts
    debug["candidate_count"] = len(candidates)

    # Deduplicate by chunk_id
    uniq = []
    seen = set()
    for d in candidates:
        cid = (d.metadata or {}).get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        uniq.append(d)

    if not uniq:
        debug["unique_count"] = 0
        print(f"[chat-retrieval] no candidates after dedupe for state={state}")
        return [], debug
    debug["unique_count"] = len(uniq)

    reranker = _qa_reranker()
    reranked = reranker.rerank(query=q, documents=uniq)[:top_k]
    debug["reranked_count"] = len(reranked)
    print(
        f"[chat-retrieval] state={state} indexes={len(indexes)} "
        f"candidates={len(candidates)} unique={len(uniq)} reranked={len(reranked)}"
    )

    out: list[dict[str, Any]] = []
    for d in reranked:
        meta = d.metadata or {}
        source = meta.get("source") or meta.get("index_name") or "unknown"
        ref = meta.get("section") or meta.get("rule") or meta.get("clause") or meta.get("chunk_id") or "UNKNOWN"
        doc_type = meta.get("doc_type") or "unknown"
        cid = meta.get("chunk_id") or ""

        snippet = " ".join((d.content or "").split())
        if len(snippet) > 420:
            snippet = snippet[:420].rstrip() + "..."

        out.append(
            {
                "source": source,
                "ref": ref,
                "doc_type": doc_type,
                "chunk_id": cid,
                "snippet": snippet,
            }
        )

    return out, debug


def _chat_context(
    question: str,
    *,
    result: Dict[str, Any],
    state: str,
    show_debug: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[dict[str, Any]]]:
    clauses = result.get("clauses", []) or []
    top = _top_relevant_clauses(question, clauses, k=4)
    retrieved, debug = _semantic_retrieve(question, state=state, top_k=6)
    return top, retrieved, (debug if show_debug else None)


@st.cache_resource(show_spinner=False)
def _qa_chat_llm() -> OllamaLegalChatAgent:
    # Defaults to Qwen3-32B via `OllamaLegalChatAgent.DEFAULT_MODEL`,
    # override with env: OLLAMA_LEGAL_CHAT_MODEL
    return OllamaLegalChatAgent()


def _chat_answer(
    question: str,
    *,
    result: Dict[str, Any],
    state: str,
    show_debug: bool = False,
) -> tuple[str, Optional[dict[str, Any]]]:
    """
    Produce a grounded, lawyer-friendly answer using ONLY the current report.
    """
    clauses = result.get("clauses", []) or []
    top = _top_relevant_clauses(question, clauses, k=4)

    # Run real semantic retrieval against indexes (system-grounded)
    retrieved, debug = _semantic_retrieve(question, state=state, top_k=6)

    lines: list[str] = []
    lines.append("Here are the most relevant report findings and grounded legal sources:")

    if top:
        lines.append("\n### Relevant clauses in this contract")
        for idx, c in enumerate(top, start=1):
            ref = c.get("normalized_reference") or f"Clause {c.get('clause_id', 'N/A')}"
            heading = c.get("heading")
            if heading:
                ref = f"{ref} - {heading}"

            alignment = c.get("alignment", "N/A")
            risk = c.get("risk_level", "N/A")
            qscore = c.get("quality_score", "N/A")
            gscore = c.get("groundedness_score", "N/A")

            statutory_refs = c.get("statutory_refs") or []
            snippet = (c.get("evidence_snippets") or [""])[0]
            citations_md = _format_citations(c.get("citations") or [])

            lines.append(f"\n**{idx}. {ref}**")
            lines.append(f"- Alignment: `{alignment}` | Risk: `{risk}` | Quality: `{qscore}` | Groundedness: `{gscore}`")
            if statutory_refs:
                joined = "; ".join(statutory_refs[:3])
                lines.append(f"- Statutory anchors: {_highlight_citations(joined)}")
            if snippet:
                lines.append(f"- Evidence snippet: {_highlight_citations(snippet)}")
            if citations_md:
                lines.append("- Citations:")
                lines.append(citations_md)

    if retrieved:
        lines.append("\n### Retrieved statutes / rules (semantic search)")
        for i, r in enumerate(retrieved, start=1):
            lines.append(
                f"\n**{i}. {_highlight_citations(str(r['source']))} — {_highlight_citations(str(r['ref']))}**"
            )
            lines.append(f"- Type: `{r.get('doc_type', 'unknown')}` | Chunk: `{r.get('chunk_id', '')}`")
            lines.append(f"- Snippet: {_highlight_citations(str(r.get('snippet', '')))}")
    else:
        lines.append(
            "\n### Retrieved statutes / rules (semantic search)\n"
            "No sources were retrieved from the legal indexes for this question. "
            "Try including a statute/rule anchor like `Section 18` or `Rule 6`."
        )

    lines.append(
        "\nIf you want, ask a follow-up like: "
        "`Show me clauses tied to Section 19`, `Why is this review_required?`, "
        "`Which clauses relate to refunds/interest?`"
    )

    return "\n".join(lines), (debug if show_debug else None)


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

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Contract Risk Analyzer")
    st.caption("Upload a contract PDF and run grounded legal-risk analysis with statute-level citations.")

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
        # Do not return: user may still want to view last report/chat.

    # Run analysis only when explicitly requested and file exists
    if analyze_clicked and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        try:
            with st.spinner("Analyzing contract... this may take a minute."):
                result = run_contract_analysis(str(temp_path), state=selected_state)
            st.success("Analysis completed.")
            st.session_state["last_result"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Show last analysis if available (persists across refresh)
    result = st.session_state.get("last_result")
    if not result:
        st.info("Upload a contract PDF and click 'Start Analyzing'.")
        return

    # Tabs for a cleaner UI
    tab_summary, tab_issues, tab_chat, tab_export = st.tabs(
        ["Summary", "Issues", "Chat", "Export"]
    )

    summary = _summarize_result(result)
    with tab_summary:
        st.markdown('<div class="cr-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", summary["overall_score"])
        col2.metric("Risk Level", summary["risk_level"])
        col3.metric("Legal Confidence", summary["legal_confidence"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Distribution")
        st.json(summary["distribution"])

        st.subheader("Grounding Diagnostics")
        grounding = _grounding_diagnostics(result)
        gd1, gd2, gd3, gd4 = st.columns(4)
        gd1.metric(
            "Avg Groundedness",
            grounding["avg_groundedness"] if grounding["avg_groundedness"] is not None else "N/A",
        )
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

    with tab_issues:
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

                st.markdown(f"**{shown}. {ref}** - {issue.get('issue', 'Issue not available')}")
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

    with tab_chat:
        st.subheader("Ask a question about this report")
        st.caption(
            "This chat answers using the current report’s clauses plus a fresh semantic search "
            "over the legal indexes (FAISS + rerank)."
        )

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        if "pending_question" not in st.session_state:
            st.session_state["pending_question"] = None
        if "last_chat_debug" not in st.session_state:
            st.session_state["last_chat_debug"] = None

        show_debug = st.checkbox("Show retrieval debug", value=False)
        use_llm_chat = st.checkbox(
            "Use lawyer-style drafted answer (LLM)",
            value=True,
            help="Uses a local LLM (Ollama) to draft a conversational lawyer-style answer grounded in retrieved RERA sources.",
        )

        with st.expander("Examples you can ask"):
            st.write("- What does **Section 19** require and which clauses relate to it?")
            st.write("- Show clauses tied to **Rule 6**.")
            st.write("- Why is the verdict **review_required**?")
            st.write("- Which clauses affect **refund / interest / possession delay**?")

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        # If a question is pending, process it *before* rendering the input widget.
        # This keeps the chat input anchored at the bottom of the chat.
        pending_question = st.session_state.get("pending_question")
        if pending_question:
            question = str(pending_question)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_Searching the RERA indexes and drafting a response…_", unsafe_allow_html=True)

                # Prepare context once; reuse for both LLM and fallback answer.
                top, retrieved, debug = _chat_context(
                    question,
                    result=result,
                    state=selected_state,
                    show_debug=show_debug,
                )
                st.session_state["last_chat_debug"] = debug

                # Default: LLM-drafted conversational answer; fallback to the older report-style output.
                answer = ""
                if use_llm_chat:
                    try:
                        placeholder.markdown("_Retrieved grounded sources. Drafting answer…_", unsafe_allow_html=True)

                        clause_ctx: list[ChatClauseContext] = []
                        for c in top:
                            display_ref = (
                                c.get("normalized_reference")
                                or f"Clause {c.get('clause_id', 'N/A')}"
                            )
                            clause_ctx.append(
                                ChatClauseContext(
                                    clause_id=str(c.get("clause_id", "")),
                                    display_ref=str(display_ref),
                                    heading=c.get("heading"),
                                    plain_summary=c.get("plain_summary"),
                                    legal_explanation=c.get("legal_explanation"),
                                    statutory_refs=list(c.get("statutory_refs") or []),
                                )
                            )

                        source_ctx: list[ChatSourceContext] = []
                        for r in retrieved:
                            source_ctx.append(
                                ChatSourceContext(
                                    source=str(r.get("source", "")),
                                    ref=str(r.get("ref", "")),
                                    doc_type=str(r.get("doc_type", "unknown")),
                                    chunk_id=str(r.get("chunk_id", "")),
                                    snippet=str(r.get("snippet", "")),
                                )
                            )

                        llm = _qa_chat_llm()
                        buf: list[str] = []
                        for delta in llm.stream_answer(
                            question,
                            state=selected_state,
                            clauses=clause_ctx,
                            sources=source_ctx,
                        ):
                            buf.append(delta)
                            placeholder.markdown(
                                _highlight_citations("".join(buf)),
                                unsafe_allow_html=True,
                            )
                        answer = _highlight_citations("".join(buf))
                        placeholder.markdown(answer, unsafe_allow_html=True)
                    except Exception as exc:
                        fallback, _ = _chat_answer(
                            question,
                            result=result,
                            state=selected_state,
                            show_debug=False,
                        )
                        answer = fallback
                        placeholder.markdown(answer, unsafe_allow_html=True)
                        st.caption(
                            f"LLM drafting unavailable; showing grounded report output instead. ({exc})"
                        )
                else:
                    placeholder.markdown("_Preparing grounded answer…_", unsafe_allow_html=True)
                    answer, _ = _chat_answer(
                        question,
                        result=result,
                        state=selected_state,
                        show_debug=False,
                    )
                    placeholder.markdown(answer, unsafe_allow_html=True)

            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            st.session_state["pending_question"] = None

        if show_debug and st.session_state.get("last_chat_debug"):
            with st.expander("Retrieval debug output"):
                st.json(st.session_state["last_chat_debug"])

        # Keep the input last so it stays at the bottom.
        new_question = st.chat_input(
            "Ask a lawyer-style question (e.g., 'Section 19 allottee rights')..."
        )
        if new_question:
            st.session_state["chat_messages"].append({"role": "user", "content": new_question})
            st.session_state["pending_question"] = new_question
            st.rerun()

    with tab_export:
        st.subheader("Export")
        st.caption("Download the full JSON report for audit and offline review.")
        output_json = json.dumps(result, indent=2)
        st.download_button(
            "Download JSON Report",
            data=output_json,
            file_name=f"contract_analysis_{selected_state}.json",
            mime="application/json",
        )

    # (analysis errors are handled above; no temp file exists here)


if __name__ == "__main__":
    main()
