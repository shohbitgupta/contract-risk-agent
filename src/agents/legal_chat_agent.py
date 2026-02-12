from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class ChatClauseContext:
    clause_id: str
    display_ref: str
    heading: str | None
    plain_summary: str | None
    legal_explanation: str | None
    statutory_refs: List[str]


@dataclass(frozen=True)
class ChatSourceContext:
    source: str
    ref: str
    doc_type: str
    chunk_id: str
    snippet: str


class OllamaLegalChatAgent:
    """
    A "legal expert" chat agent that synthesizes a lawyer-style response
    grounded ONLY in:
    - relevant analyzed clause outputs (report context)
    - retrieved statutory/rule snippets from the RERA indexes

    Uses local Ollama for generation and supports streaming output.
    """

    # Recommended: a stronger instruction-following model for legal drafting.
    # Override with env var if needed:
    #   export OLLAMA_LEGAL_CHAT_MODEL="qwen3:32b"
    DEFAULT_MODEL = os.getenv("OLLAMA_LEGAL_CHAT_MODEL", "llama3:8b")

    def __init__(self, *, model: Optional[str] = None) -> None:
        self.model = (model or self.DEFAULT_MODEL).strip()

    def stream_answer(
        self,
        question: str,
        *,
        state: str,
        clauses: List[ChatClauseContext],
        sources: List[ChatSourceContext],
    ) -> Iterator[str]:
        """
        Stream the assistant answer as text chunks.
        """
        if not question or not question.strip():
            return iter(())

        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama is not available on PATH. Install Ollama or disable LLM chat."
            )

        prompt = self._build_prompt(
            question=question,
            state=state,
            clauses=clauses,
            sources=sources,
        )

        proc = subprocess.Popen(
            ["ollama", "run", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()

        # Stream bytes in chunks; Ollama flushes progressively.
        try:
            while True:
                chunk = proc.stdout.read(256)
                if not chunk:
                    break
                yield chunk.decode("utf-8", errors="ignore")
        finally:
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"Ollama chat failed (exit={rc}): {err}")

    def _build_prompt(
        self,
        *,
        question: str,
        state: str,
        clauses: List[ChatClauseContext],
        sources: List[ChatSourceContext],
    ) -> str:
        clause_lines: List[str] = []
        for i, c in enumerate(clauses, start=1):
            heading = f" — {c.heading}" if c.heading else ""
            clause_lines.append(f"[C{i}] {c.display_ref}{heading}")
            if c.plain_summary:
                clause_lines.append(f"Plain: {c.plain_summary}")
            if c.legal_explanation:
                clause_lines.append(f"Legal: {c.legal_explanation}")
            if c.statutory_refs:
                clause_lines.append("Anchors: " + "; ".join(c.statutory_refs[:4]))
            clause_lines.append("")

        source_lines: List[str] = []
        for i, s in enumerate(sources, start=1):
            source_lines.append(
                f"[S{i}] {s.source} — {s.ref} (type={s.doc_type}, chunk={s.chunk_id})"
            )
            if s.snippet:
                source_lines.append(f"Snippet: {s.snippet}")
            source_lines.append("")

        clauses_block = "\n".join(clause_lines).strip() or "None."
        sources_block = "\n".join(source_lines).strip() or "None."

        return textwrap.dedent(
            f"""
            You are a senior Indian real estate lawyer specializing in RERA compliance ({state}).

            Your job: answer the user's question in a lawyer-drafted, conversational, human-readable way,
            grounded ONLY in the provided sources and contract-report context.

            HARD RULES (non-negotiable):
            - Use ONLY the text in SOURCES and CONTRACT CONTEXT. Do NOT invent or assume missing law.
            - If the sources are insufficient, say so and explain what is missing.
            - Do not give absolute legal advice; provide a risk-aware, informational response.
            - Avoid dumping sections. Explain what the section means in practice and how it impacts the clause.
            - When you make a claim, cite it inline using [S#] and/or [C#].
            - Do NOT cite any source that is not listed below.

            OUTPUT FORMAT (Markdown):
            - Start with a short, direct answer (2–5 sentences).
            - Then provide a reasoned explanation in plain English (lawyer tone).
            - Include a "Practical next steps" section.
            - If relevant, include "Suggested drafting" with 3–8 lines of sample clause wording.
            - End with "Sources cited" listing the [S#] and [C#] you used.

            USER QUESTION:
            {question.strip()}

            CONTRACT CONTEXT (from this report):
            {clauses_block}

            SOURCES (retrieved from RERA indexes):
            {sources_block}
            """
        ).strip()

