import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from agents.llm_analyzer_facade import LegalLLMFacade
from tools.llm_response_cache import LLMResponseCache

from RAG.models import (
    ClauseUnderstandingResult,
    EvidencePack,
    ExplanationResult
)
from RAG.user_contract_chunker import ContractChunk
from tools.logger import setup_logger

logger = setup_logger("legal-explanation-agent")


class LegalExplanationAgent:
    """
    Evidence-bound, guardrailed legal explanation agent.
    """

    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.0

    SYSTEM_PROMPT = """
    You are a legal explanation engine.

    HARD CONSTRAINTS:
    - Use ONLY the evidence provided.
    - Do NOT invent laws, rules, or interpretations.
    - Do NOT give legal advice.
    - If evidence is insufficient, explicitly say so.
    - Every claim MUST reference an evidence_id.
    - Output MUST be valid JSON matching the schema.
    

    ALLOWED ALIGNMENT VALUES:
    - aligned
    - partially_aligned
    - conflicting
    - insufficient_evidence
    """

    OUTPUT_SCHEMA = {
        "alignment": "string",
        "key_findings": "array",
        "explanation": "string",
        "evidence_mapping": "array"
    }

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Set it in .env or environment variables."
            )

        # OpenAI client (used by refiner inside facade)
        self.client = OpenAI(api_key=api_key)

        # LLM facade (Local → OpenAI refine)
        self.llm_facade = LegalLLMFacade()

        # Deterministic cache for final explanations
        self.cache = LLMResponseCache(
            cache_dir=Path("data/llm_cache")
        )

    def explain(
        self,
        clause: ContractChunk,
        clause_result: ClauseUnderstandingResult,
        evidence_pack: EvidencePack
    ) -> ExplanationResult:

        cache_key = self.cache.build_cache_key(
            clause_text=clause.text,
            intent=clause_result.intent,
            obligation_type=clause_result.obligation_type,
            evidence_pack=evidence_pack
        )

        cached = self.cache.get(cache_key)

        if cached:
            logger.info("LLM cache hit")
        else:
            logger.info("LLM cache miss")

        if cached:
            return ExplanationResult(**cached)

        # Build evidence text for facade
        evidence_text = self._build_evidance(clause, clause_result, evidence_pack)

        logger.info(f"LLM evidence text: {evidence_text}")

        # Delegate execution to LLM facade (local → OpenAI)
        raw_output = self.llm_facade.explain(
            clause_text=clause.text,
            evidence_text=evidence_text
        )

        logger.info(f"Raw output: {raw_output}")

        parsed = self._parse_and_validate_output(raw_output, evidence_pack)

        quality_score = self._score_explanation(parsed, evidence_pack)

        return ExplanationResult(
            clause_id=clause.chunk_id,
            alignment=self._determine_alignment(clause_result, evidence_pack),
            risk_level=clause_result.risk_level,
            summary=parsed["key_findings"][0],
            detailed_explanation=parsed["explanation"],
            citations=self._build_citations(evidence_pack),
            quality_score=quality_score,
            disclaimer=self._disclaimer()
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_evidance(
        self,
        clause: ContractChunk,
        clause_result: ClauseUnderstandingResult,
        evidence_pack: EvidencePack
    ) -> str:

        evidence_block = ""
        for i, ev in enumerate(evidence_pack.evidences, start=1):
            evidence_block += (
                f"[Evidence {i}]\n"
                f"Source: {ev.source}\n"
                f"Section: {ev.section_or_clause}\n"
                f"Text: {ev.text}\n\n"
            )

        return f"""
USER CONTRACT CLAUSE:
Clause ID: {clause.chunk_id}
Text:
{clause.text}

DETECTED INTENT:
{clause_result.intent}

RISK LEVEL:
{clause_result.risk_level}

LEGAL EVIDENCE:
{evidence_block}
"""

    # ------------------------------------------------------------------
    # Guardrails & Validation
    # ------------------------------------------------------------------

    def _parse_and_validate_output(
        self,
        output: str,
        evidence_pack: EvidencePack
    ) -> Dict:

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            raise ValueError("LLM output is not valid JSON")

        if parsed.get("alignment") not in {
            "aligned", "partially_aligned", "conflicting", "insufficient_evidence"
        }:
            raise ValueError("Invalid alignment value")

        evidence_ids = {
            f"Evidence {i+1}"
            for i in range(len(evidence_pack.evidences))
        }

        for mapping in parsed.get("evidence_mapping", []):
            if mapping.get("evidence_id") not in evidence_ids:
                raise ValueError("Invalid or hallucinated evidence reference")

        return parsed

    # ------------------------------------------------------------------
    # Quality Scoring (Deterministic)
    # ------------------------------------------------------------------

    def _score_explanation(
        self,
        parsed: Dict,
        evidence_pack: EvidencePack
    ) -> float:

        score = 0.0

        # Evidence coverage
        if len(evidence_pack.evidences) >= 2:
            score += 0.3

        # Authority strength
        if any(ev.metadata["doc_type"] == "rera_act" for ev in evidence_pack.evidences):
            score += 0.3
        elif any(ev.metadata["doc_type"] == "state_rule" for ev in evidence_pack.evidences):
            score += 0.2

        # Jurisdiction correctness
        score += 0.2

        # Uncertainty honesty
        if parsed["alignment"] == "insufficient_evidence":
            score += 0.2

        return round(min(score, 1.0), 2)

    # ------------------------------------------------------------------

    def _build_citations(self, evidence_pack: EvidencePack) -> List[Dict]:
        return [
            {
                "source": ev.source,
                "section_or_clause": ev.section_or_clause
            }
            for ev in evidence_pack.evidences
        ]

    def _disclaimer(self) -> str:
        return (
            "This explanation is for informational purposes only and does not "
            "constitute legal advice. Consult a qualified legal professional."
        )

    def _determine_alignment(
            self,
            clause_result: ClauseUnderstandingResult,
            evidence_pack: EvidencePack
    ) -> str:
        if not evidence_pack.evidences:
            return "insufficient_evidence"

        if clause_result.risk_level == "high":
            return "conflicting"

        return "partially_aligned"
