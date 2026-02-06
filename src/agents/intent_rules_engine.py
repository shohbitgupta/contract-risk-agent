import yaml
from pathlib import Path
from typing import Dict, Any, List

from RAG.models import ClauseUnderstandingResult
from utils.schema_factory import build_model
from utils.schema_drift import log_schema_drift
from configs.schema_config import STRICT_SCHEMA


class IntentRuleEngine:
    """
    Deterministic intent detection engine for Indian real estate contracts
    aligned with RERA (Central + State).
    """

    def __init__(self, rules_path: Path):
        if not rules_path.exists():
            raise FileNotFoundError(f"Intent rules file not found: {rules_path}")

        with open(rules_path, "r") as f:
            self.rules = yaml.safe_load(f)

        self.base_intents = self.rules.get("intents", {})
        self.violation_intents = self.rules.get("violation_only_intents", {})
        self.state_overrides = self.rules.get("state_overrides", {})
        self.global_cfg = self.rules.get("global", {})

        self.implicit_markers = [
            m.lower() for m in self.global_cfg.get("implicit_compliance_markers", [])
        ]

        self.default_risk = self.global_cfg.get(
            "default_risk_if_uncertain", "medium"
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        clause_id: str,
        clause_text: str,
        state: str | None = None
    ) -> ClauseUnderstandingResult:

        text = clause_text.lower()

        # -----------------------------------------------------
        # 1️⃣ Violation-only intents (highest priority)
        # -----------------------------------------------------
        violation = self._match_violation_intent(text)
        if violation:
            return self._build_violation_result(
                clause_id=clause_id,
                violation_cfg=violation
            )

        # -----------------------------------------------------
        # 2️⃣ Base intent detection
        # -----------------------------------------------------
        intent_key, intent_cfg = self._match_base_intent(text)

        if not intent_key:
            # Conservative fallback
            data = {
                "clause_id": clause_id,
                "intent": "unknown",
                "obligation_type": "unclear",
                "risk_level": self.default_risk,
                "needs_legal_validation": True,
                "retrieval_queries": [],
                "compliance_mode": "UNKNOWN",
                "compliance_confidence": 0.0,
                "notes": ["No matching intent rule found"],
            }

            return build_model(
                ClauseUnderstandingResult,
                data,
                strict=STRICT_SCHEMA,
                log_fn=log_schema_drift
            )

        # -----------------------------------------------------
        # 3️⃣ Apply state overrides
        # -----------------------------------------------------
        effective_cfg = self._apply_state_override(
            intent_key=intent_key,
            base_cfg=intent_cfg,
            state=state
        )

        # -----------------------------------------------------
        # 4️⃣ Compliance mode
        # -----------------------------------------------------
        compliance_mode = self._detect_compliance_mode(text)

        # -----------------------------------------------------
        # 5️⃣ Risk level
        # -----------------------------------------------------
        risk_level = self._determine_risk(text, effective_cfg)

        # -----------------------------------------------------
        # 6️⃣ Obligation type
        # -----------------------------------------------------
        obligation_type = effective_cfg.get(
            "obligation_type", "promoter"
        )

        # -----------------------------------------------------
        # 7️⃣ Legal validation requirement
        # -----------------------------------------------------
        needs_legal_validation = (
            risk_level != "low"
            or compliance_mode == "UNKNOWN"
        )

        # -----------------------------------------------------
        # 8️⃣ Retrieval queries
        # -----------------------------------------------------
        retrieval_queries = self._build_retrieval_queries(
            intent_key=intent_key,
            intent_cfg=effective_cfg
        )

        data = {
            "clause_id": clause_id,
            "intent": effective_cfg.get("intent_name", intent_key),
            "obligation_type": obligation_type,
            "risk_level": risk_level,
            "needs_legal_validation": needs_legal_validation,
            "retrieval_queries": retrieval_queries,
            "compliance_mode": compliance_mode,
            "compliance_confidence": 0.0,  # filled later by ClauseUnderstandingAgent
            "notes": [],
        }

        return build_model(
            ClauseUnderstandingResult,
            data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    # =========================================================
    # VIOLATION INTENTS
    # =========================================================

    def _match_violation_intent(self, text: str) -> Dict[str, Any] | None:
        for cfg in self.violation_intents.values():
            for kw in cfg.get("keywords", []):
                if kw.lower() in text:
                    return cfg
        return None

    def _build_violation_result(
        self,
        clause_id: str,
        violation_cfg: Dict[str, Any]
    ) -> ClauseUnderstandingResult:

        retrieval_cfg = violation_cfg.get("retrieval", {})

        data = {
            "clause_id": clause_id,
            "intent": violation_cfg["intent_name"],
            "obligation_type": "promoter",
            "risk_level": violation_cfg.get("risk_level", "high"),
            "needs_legal_validation": True,
            "retrieval_queries": [
                {"index": idx, "filters": retrieval_cfg.get("filters", {})}
                for idx in retrieval_cfg.get("indexes", [])
            ],
            "compliance_mode": "CONTRADICTION",
            "compliance_confidence": 0.0,
            "notes": ["Violation-only intent detected"],
        }

        return build_model(
            ClauseUnderstandingResult,
            data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    # =========================================================
    # BASE INTENT MATCHING
    # =========================================================

    def _match_base_intent(self, text: str):
        for intent_key, cfg in self.base_intents.items():
            for kw in cfg.get("keywords", []):
                if kw.lower() in text:
                    return intent_key, cfg
        return None, None

    # =========================================================
    # STATE OVERRIDES
    # =========================================================

    def _apply_state_override(
        self,
        intent_key: str,
        base_cfg: Dict[str, Any],
        state: str | None
    ) -> Dict[str, Any]:

        if not state:
            return base_cfg

        state_cfg = self.state_overrides.get(state.lower())
        if not state_cfg:
            return base_cfg

        override = state_cfg.get(intent_key)
        if not override:
            return base_cfg

        merged = dict(base_cfg)
        merged.update(override)
        return merged

    # =========================================================
    # COMPLIANCE MODE
    # =========================================================

    def _detect_compliance_mode(self, text: str) -> str:
        for marker in self.implicit_markers:
            if marker in text:
                return "IMPLICIT"
        return "UNKNOWN"

    # =========================================================
    # RISK DETERMINATION
    # =========================================================

    def _determine_risk(
        self,
        text: str,
        intent_cfg: Dict[str, Any]
    ) -> str:

        rules = intent_cfg.get("risk_rules", {})

        for level in ("high", "medium", "low"):
            for phrase in rules.get(level, []):
                if phrase.lower() in text:
                    return level

        return self.default_risk

    # =========================================================
    # RETRIEVAL QUERY BUILDER
    # =========================================================

    def _build_retrieval_queries(
        self,
        intent_key: str,
        intent_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        retrieval = intent_cfg.get("retrieval", {})
        indexes = retrieval.get("indexes", [])
        filters = retrieval.get("filters", {})

        return [
            {
                "index": idx,
                "intent": intent_key,
                "filters": filters
            }
            for idx in indexes
        ]
