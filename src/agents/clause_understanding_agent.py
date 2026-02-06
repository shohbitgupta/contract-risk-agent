from pathlib import Path

from RAG.models import (
    ClauseUnderstandingResult
)

from agents.intent_rules_engine import IntentRuleEngine
from RAG.user_contract_chunker import ContractChunk


class ClauseUnderstandingAgent:
    """
    Clause understanding layer that maps a clause to a legal intent.

    Responsibilities:
    - Accept a parsed contract chunk
    - Determine the legal intent using IntentRuleEngine
    - Emit retrieval instructions for downstream retrieval

    This agent:
    - DOES NOT select indexes
    - DOES NOT apply legal reasoning
    - DOES NOT modify retrieval queries
    """

    def __init__(self, rules_path: Path):
        self.intent_engine = IntentRuleEngine(rules_path)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def analyze(
        self,
        clause: ContractChunk,
        state: str
    ) -> ClauseUnderstandingResult:
        """
        Analyze a single contract clause and return structured intent output.

        Parameters:
        - clause: ContractChunk (id + text)
        - state: Jurisdiction (currently unused here, passed downstream)

        Returns:
        - ClauseUnderstandingResult
        """

        # 🔹 Detect implicit legal incorporation
        text_lower = clause.text.lower()
        implicit_markers = [
            "as per the act",
            "as per rera",
            "in accordance with the act",
            "subject to the act",
            "subject to rules",
            "as prescribed under"
        ]

        if any(m in text_lower for m in implicit_markers):
            compliance_mode = "IMPLICIT"
        else:
            compliance_mode = "UNKNOWN"

        result.compliance_mode = compliance_mode
        return result

        return result
