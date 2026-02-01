import subprocess
import textwrap


class LocalLLMAdapter:
    """
    Local LLM adapter using Llama 3.1 8B via Ollama.
    Generates a structured draft explanation.

    Example:
        >>> adapter = LocalLLMAdapter()
        >>> adapter.generate_draft("Clause text", "Evidence text")
        'Draft explanation...'
    """

    MODEL = "llama3.1:8b"

    def generate_draft(
        self,
        clause_text: str,
        evidence_text: str
    ) -> str:
        """
        Generate a first-pass legal explanation draft.
        No final compliance judgment.

        Returns:
            Draft text (not guaranteed to be JSON).

        Example:
            >>> adapter.generate_draft("Delay clause", "Evidence 1...")
            'The clause indicates...'
        """

        prompt = textwrap.dedent(f"""
        You are a legal assistant helping analyze Indian real estate contracts.

        STRICT RULES:
        - Use ONLY the provided evidence
        - Do NOT invent laws or clauses
        - Do NOT give final legal conclusions
        - Write a structured draft explanation

        CONTRACT CLAUSE:
        ----------------
        {clause_text}

        LEGAL EVIDENCE:
        ----------------
        {evidence_text}

        DRAFT EXPLANATION:
        """)

        result = subprocess.run(
            ["ollama", "run", self.MODEL],
            input=prompt,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Local LLM error: {result.stderr}"
            )

        return result.stdout.strip()
