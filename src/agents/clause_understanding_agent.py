from pathlib import Path

from RAG.models import (
    ClauseUnderstandingResult
)

from agents.intent_rules_engine import IntentRuleEngine
from RAG.user_contract_chunker import ContractChunk


class ClauseUnderstandingAgent:
    """
    Clause understanding layer.

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
        Analyze a single contract clause.

        Parameters:
        - clause: ContractChunk (id + text)
        - state: Jurisdiction (currently unused here, passed downstream)

        Returns:
        - ClauseUnderstandingResult
        """

        # Delegate intent resolution completely to rule engine
        result = self.intent_engine.analyze(
            clause_id=clause.chunk_id,
            clause_text=clause.text
        )

        return result
