import json
import re
import subprocess
from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator


class OpenAIRefiner:
    """
    Refines and validates the draft explanation using OpenAI.

    This class currently runs a local Ollama model to produce
    strict JSON output from a draft explanation.

    Example:
        >>> refiner = OpenAIRefiner(api_key="")
        >>> refiner.refine("draft...", "clause...", "evidence...")
        '{"alignment": "aligned", ...}'
    """

    MODEL = "llama3:8b"

    class EvidenceMapping(BaseModel):
        """
        Mapping between a claim and its evidence identifier.
        """
        claim: str
        evidence_id: str

    class LLMOutput(BaseModel):
        """
        Normalized LLM output schema expected by the explanation agent.
        """
        alignment: str | None = None
        key_findings: List[str] = Field(default_factory=list)
        explanation: str | None = None
        evidence_mapping: List["OpenAIRefiner.EvidenceMapping"] = Field(default_factory=list)

        @field_validator("alignment", mode="before")
        def normalize_alignment(cls, value):
            text = str(value or "").lower().strip()
            mapping = {
                "aligned": "aligned",
                "partial": "partially_aligned",
                "partially aligned": "partially_aligned",
                "partially_aligned": "partially_aligned",
                "conflict": "conflicting",
                "conflicting": "conflicting",
                "insufficient": "insufficient_evidence",
                "insufficient evidence": "insufficient_evidence",
                "insufficient_evidence": "insufficient_evidence"
            }
            return mapping.get(text, text) if text else ""

        @field_validator("key_findings", mode="before")
        def normalize_key_findings(cls, value) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            return value

    def __init__(self, api_key: str):
        # API key not required for local Ollama model.
        pass

    def refine(
        self,
        draft_text: str,
        clause_text: str,
        evidence_text: str
    ) -> str:
        """
        Refine a draft explanation into strict JSON.

        Returns:
            JSON string matching the explanation schema.

        Example:
            >>> refiner.refine("draft", "clause", "evidence")
            '{"alignment": "aligned", ...}'
        """
        prompt = f"""
You are a senior Indian real estate legal expert specializing in RERA compliance.

TASK:
1. Review the draft explanation for accuracy against the legal evidence
2. Correct any factual inaccuracies or misinterpretations
3. Use only the provided legal evidence - DO NOT introduce new legal sources
4. Return structured JSON that matches the required schema

OUTPUT SCHEMA (JSON):
{{
  "alignment": "aligned | partially_aligned | conflicting | insufficient_evidence",
  "key_findings": ["short bullet-like sentences"],
  "explanation": "plain text explanation",
  "evidence_mapping": [
    {{
      "claim": "statement in the explanation",
      "evidence_id": "Evidence 1"
    }}
  ]
}}

RULES:
- Use ONLY evidence IDs that appear in the LEGAL EVIDENCE section (e.g., "Evidence 1").
- If evidence is insufficient, set alignment to "insufficient_evidence" and explain why.
- Return ONLY valid JSON. No markdown, no extra text.

CONTRACT CLAUSE:
{clause_text}

LEGAL EVIDENCE:
{evidence_text}

DRAFT EXPLANATION (TO REFINE):
{draft_text}
"""

        response = subprocess.run(
            ["ollama", "run", self.MODEL],
            input=prompt,
            text=True,
            capture_output=True
        )

        if response.returncode != 0:
            raise RuntimeError(
                f"Ollama refiner error: {response.stderr}"
            )

        raw = response.stdout.strip()
        try:
            return self._extract_json(raw)
        except ValueError:
            # One retry with a stricter JSON-only prompt
            retry_prompt = f"""
Return ONLY valid JSON for the following output. Do not add any text.

OUTPUT TO FIX:
{raw}
"""
            retry = subprocess.run(
                ["ollama", "run", self.MODEL],
                input=retry_prompt,
                text=True,
                capture_output=True
            )
            if retry.returncode != 0:
                raise RuntimeError(
                    f"Ollama refiner retry error: {retry.stderr}"
                )
            return self._extract_json(retry.stdout.strip())

    def _extract_json(self, text: str) -> str:
        """
        Best-effort JSON extraction for strict output handling.

        The method supports:
        - direct JSON
        - JSON embedded in extra text
        - minor fixes (trailing commas)
        """
        text = text.strip()
        try:
            parsed = json.loads(text)
            return self._normalize_output(parsed)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            sanitized = candidate.replace("\t", " ")
            sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
            parsed = json.loads(sanitized)
            return self._normalize_output(parsed)

        raise ValueError("Refiner did not return valid JSON")

    def _normalize_output(self, parsed: dict) -> str:
        """
        Normalize fields to match expected schema values.

        Returns:
            A JSON string after Pydantic validation and normalization.
        """
        try:
            model = self.LLMOutput.model_validate(parsed)
        except ValidationError:
            # Best-effort fallback if model returns a nested/unknown schema.
            model = self.LLMOutput(
                alignment="insufficient_evidence",
                key_findings=["Unable to parse model output reliably."],
                explanation=(
                    "The explanation output did not match the required schema. "
                    "Please retry or inspect the raw model output."
                ),
                evidence_mapping=[]
            )

        data = model.model_dump()
        if not data.get("alignment"):
            data["alignment"] = "insufficient_evidence"
        if not data.get("explanation"):
            data["explanation"] = "No explanation provided."
        if not data.get("key_findings"):
            data["key_findings"] = ["No key findings provided."]

        return json.dumps(data, ensure_ascii=False)
