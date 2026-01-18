import subprocess

from openai import OpenAI


class OpenAIRefiner:
    """
    Refines and validates the draft explanation using OpenAI.
    """

    MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def refine(
        self,
        draft_text: str,
        clause_text: str,
        evidence_text: str
    ) -> str:
        prompt = f"""
        You are a senior Indian real estate legal expert specializing in RERA compliance.

        TASK:
        1. Review the draft explanation for accuracy against the legal evidence
        2. Correct any factual inaccuracies or misinterpretations
        3. Ensure strict RERA (Real Estate Regulation Act) compliance
        4. Keep it concise, precise, and professional
        5. Use only the provided legal evidence - DO NOT introduce new legal sources
        6. Format as plain text paragraphs (no markdown, no bullet points)

        INPUT FORMAT:
        - Contract Clause: The actual contract text to analyze
        - Legal Evidence: Supporting RERA regulations/case law
        - Draft: Initial explanation to refine

        CONTRACT CLAUSE:
        {clause_text}

        LEGAL EVIDENCE:
        {evidence_text}

        DRAFT EXPLANATION (TO REFINE):
        {draft_text}

        OUTPUT REQUIREMENTS:
        - Return ONLY the refined explanation text
        - No introductory phrases like "Here is" or "The refined explanation is"
        - No disclaimers or meta-commentary
        - Directly start with the refined content

        REFINED EXPLANATION:
        """

        response = subprocess.run(
            ["ollama", "run", self.MODEL],
            input=prompt,
            text=True,
            capture_output=True
        )

        return response.stdout.strip()
