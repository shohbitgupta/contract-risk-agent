import os

from agents.legal_details_drafter_agent import LocalLLMAdapter
from agents.legal_details_verifier_agent import OpenAIRefiner


class LegalLLMFacade:
    """
    Two-stage LLM explanation pipeline.

    Stage 1: Local draft (fast, cheap, may be imperfect)
    Stage 2: Refiner (enforces structure and correctness)

    Example:
        >>> facade = LegalLLMFacade()
        >>> facade.explain("Clause text...", "Evidence text...")
        '{"alignment": "...", "key_findings": [...], ...}'
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.local_llm = LocalLLMAdapter()
        self.refiner = OpenAIRefiner(api_key)

    def explain(
        self,
        clause_text: str,
        evidence_text: str
    ) -> str:
        """
        Generate a structured explanation by chaining draft + refiner.

        Returns:
            JSON string output that matches the required schema.

        Example:
            >>> facade.explain("Delay clause", "Evidence 1...")
            '{"alignment": "aligned", ...}'
        """
        # Stage 1: Local draft
        draft = self.local_llm.generate_draft(
            clause_text=clause_text,
            evidence_text=evidence_text
        )

        # Stage 2: OpenAI refinement
        return self.refiner.refine(
            draft_text=draft,
            clause_text=clause_text,
            evidence_text=evidence_text
        )
