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
    - RERA-aware intent detection
    - IMPLICIT / EXPLICIT / CONTRADICTION compliance handling
    - Deterministic compliance confidence scoring

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
        Analyze a contract clause and enrich it with
        legal intent, risk, compliance mode, and confidence.
        """

        # 1️⃣ Run intent rules engine
        result = self.intent_engine.analyze(
            clause_id=clause.chunk_id,
            clause_text=clause.text,
            state=state
        )

        # 2️⃣ Compute compliance confidence
        confidence = self._compute_compliance_confidence(
            clause=clause,
            result=result
        )

        # 3️⃣ Attach confidence
        result.compliance_confidence = confidence

        return result

    # -----------------------------------------------------
    # Confidence Scoring Logic
    # -----------------------------------------------------

    def _compute_compliance_confidence(
        self,
        clause: ContractChunk,
        result: ClauseUnderstandingResult
    ) -> float:
        """
        Deterministic confidence scoring for legal interpretation.

        Output range: 0.0 – 1.0
        """

        score = 0.5  # neutral baseline

        # Intent clarity
        if result.intent and result.intent != "unknown":
            score += 0.2
        else:
            score -= 0.2

        # Compliance mode
        if result.compliance_mode in ("IMPLICIT", "EXPLICIT"):
            score += 0.2
        elif result.compliance_mode == "CONTRADICTION":
            score -= 0.3

        # Risk signal clarity
        if result.risk_level in ("high", "medium", "low"):
            score += 0.1

        # Structural confidence from chunker
        if getattr(clause, "confidence", 0) >= 0.8:
            score += 0.1

        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        return round(score, 2)
