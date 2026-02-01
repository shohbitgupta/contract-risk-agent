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

    Example:
        >>> agent = ClauseUnderstandingAgent(Path("src/configs/real_state_intent_rules.yaml"))
        >>> result = agent.analyze(ContractChunk("1.1", "Delay in possession...", ChunkType.CLAUSE), "uttar_pradesh")
        >>> result.intent
        'possession_delay'
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

        Example:
            >>> clause = ContractChunk("2.1", "Refund on cancellation...", ChunkType.CLAUSE)
            >>> agent.analyze(clause, "uttar_pradesh").intent
            'refund_and_withdrawal'
        """

        # Delegate intent resolution completely to rule engine
        result = self.intent_engine.analyze(
            clause_id=clause.chunk_id,
            clause_text=clause.text
        )

        return result
