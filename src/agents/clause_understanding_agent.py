from pathlib import Path

from RAG.models import ClauseUnderstandingResult
from RAG.user_contract_chunker import ContractChunk
from agents.intent_rules_engine import IntentRuleEngine


class ClauseUnderstandingAgent:
    """
    Clause understanding layer that maps a clause to a legal intent.

    Responsibilities:
    - RERA-aware intent detection
    - IMPLICIT / EXPLICIT / CONTRADICTION compliance handling
    - Deterministic semantic confidence scoring

    This agent:
    - DOES NOT select indexes
    - DOES NOT apply legal reasoning
    - DOES NOT modify retrieval queries
    """

    # -------------------------------------------------
    # Init
    # -------------------------------------------------

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
        Analyze a contract clause and enrich it with:
        - legal intent
        - compliance mode
        - statutory basis
        - semantic confidence
        """

        # 1️⃣ Run intent rules engine
        result = self.intent_engine.analyze(
            clause_id=clause.chunk_id,
            clause_text=clause.text,
            state=state
        )

        # 2️⃣ Compute compliance confidence
        compliance_confidence = self._compute_compliance_confidence(
            clause=clause,
            result=result
        )

        # 2️⃣ Compute semantic confidence
        confidence = self._compute_semantic_confidence(
            clause=clause,
            result=result
        )

        # 3️⃣ Attach confidence (STRICT-safe)
        result = result.model_copy(
            update={
                "compliance_confidence": compliance_confidence,
                "semantic_confidence": getattr(clause, "semantic_confidence", None),
            }
        )

        return result

    # -------------------------------------------------
    # Semantic Confidence Scoring
    # -------------------------------------------------

    def _compute_semantic_confidence(
        self,
        clause: ContractChunk,
        result: ClauseUnderstandingResult
    ) -> float:
        """
        Deterministic, lawyer-defensible confidence scoring.

        Interpretation:
        - High score ≠ good clause
        - High score = high certainty of legal interpretation

        Output range: 0.0 – 1.0
        """

        score = 0.45  # conservative neutral baseline

        # ---------------------------------------------
        # 1️⃣ Intent clarity
        # ---------------------------------------------
        if result.intent and result.intent != "unknown":
            score += 0.2
        else:
            score -= 0.15

        # ---------------------------------------------
        # 2️⃣ Compliance mode certainty
        # ---------------------------------------------
        if result.compliance_mode == "EXPLICIT":
            score += 0.2
        elif result.compliance_mode == "IMPLICIT":
            score += 0.1
        elif result.compliance_mode == "CONTRADICTION":
            # Illegality = high certainty
            score = max(score, 0.8)
        else:  # UNKNOWN
            score -= 0.1

        # ---------------------------------------------
        # 3️⃣ Statutory anchoring
        # ---------------------------------------------
        statutory = getattr(result, "statutory_basis", None)
        if statutory and statutory.get("sections"):
            score += 0.15
        else:
            score -= 0.05

        # ---------------------------------------------
        # 4️⃣ Risk signal clarity
        # ---------------------------------------------
        if result.risk_level in ("high", "medium", "low"):
            score += 0.05

        # ---------------------------------------------
        # 5️⃣ Structural confidence from chunker
        # ---------------------------------------------
        if getattr(clause, "confidence", 0.0) >= 0.8:
            score += 0.1
        elif getattr(clause, "confidence", 0.0) < 0.5:
            score -= 0.1

        # ---------------------------------------------
        # Clamp
        # ---------------------------------------------
        score = max(0.0, min(1.0, score))
        return round(score, 2)


    def _compute_compliance_confidence(self, clause, result) -> float:
        score = 0.5

        if result.intent != "unknown":
            score += 0.2
        if result.compliance_mode in ("IMPLICIT", "EXPLICIT"):
            score += 0.2
        if result.compliance_mode == "CONTRADICTION":
            score -= 0.3
        if result.risk_level in ("high", "medium", "low"):
            score += 0.1
        if clause.confidence >= 0.8:
            score += 0.1

        return round(max(0.0, min(1.0, score)), 2)
