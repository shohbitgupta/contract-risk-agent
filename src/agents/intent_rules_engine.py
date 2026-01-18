from pathlib import Path
from typing import Dict, Any, List
import yaml

from RAG.models import ClauseUnderstandingResult


class IntentRuleEngine:
    """
    YAML-driven intent resolution engine.

    Responsibilities:
    - Match clause text to intent
    - Classify obligation type (high-level)
    - Emit retrieval instructions
    - Provide safe defaults for downstream legal analysis
    """

    def __init__(self, rules_path: Path):
        if not rules_path.exists():
            raise FileNotFoundError(
                f"Intent rules file not found: {rules_path}"
            )

        with open(rules_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "intents" not in raw:
            raise ValueError("Invalid intent rules YAML")

        self.intents: Dict[str, Dict[str, Any]] = raw["intents"]

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def analyze(
        self,
        clause_id: str,
        clause_text: str
    ) -> ClauseUnderstandingResult:
        """
        Analyze clause and emit structured understanding.
        """

        matched = self._match_intent(clause_text)

        retrieval_queries = self._build_retrieval_queries(
            clause_text=clause_text,
            intent_cfg=matched["config"]
        )

        obligation_type = self._infer_obligation_type(
            intent_name=matched["name"]
        )

        return ClauseUnderstandingResult(
            clause_id=clause_id,
            intent=matched["name"],
            obligation_type=obligation_type,
            risk_level="UNKNOWN",                 # resolved later
            needs_legal_validation=True,          # always true here
            retrieval_queries=retrieval_queries
        )

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _match_intent(self, clause_text: str) -> Dict[str, Any]:
        """
        Keyword-based intent matching.
        First match wins.
        """

        text = clause_text.lower()

        for intent_name, cfg in self.intents.items():
            for kw in cfg.get("keywords", []):
                if kw.lower() in text:
                    return {
                        "name": intent_name,
                        "config": cfg
                    }

        # Safe fallback
        return {
            "name": "agreement_terms",
            "config": self.intents["agreement_terms"]
        }

    def _build_retrieval_queries(
        self,
        clause_text: str,
        intent_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build retrieval queries directly from YAML.
        """

        retrieval = intent_cfg.get("retrieval")
        if not retrieval:
            raise ValueError("Missing retrieval config in intent")

        return [{
            "index": retrieval["index"],
            "intent": clause_text,
            "filters": retrieval.get("filters", {})
        }]

    def _infer_obligation_type(self, intent_name: str) -> str:
        """
        Lightweight obligation classification.
        This is NOT legal reasoning.
        """

        if intent_name in {
            "possession_delay",
            "refund_and_withdrawal",
            "defect_liability",
            "interest_and_compensation"
        }:
            return "PROMOTER_OBLIGATION"

        if intent_name in {
            "maintenance_and_common_areas"
        }:
            return "SHARED_OBLIGATION"

        return "CONTRACTUAL_TERM"
