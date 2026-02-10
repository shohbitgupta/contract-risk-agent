from typing import Optional, Tuple, List, Dict, Any

from tools.logger import setup_logger
from RAG.contract_analysis import ClauseAnalysisResult

from utils.schema_factory import build_model
from utils.schema_drift import log_schema_drift
from utils.statute_normalizer import normalize_statutory_basis
from configs.schema_config import STRICT_SCHEMA

logger = setup_logger("legal-explanation-agent")


class LegalExplanationAgent:
    """
    Generates legally grounded, lawyer-grade explanations for a clause.

    Responsibilities:
    - Determine stance and alignment
    - Produce tiered explanations (plain + legal)
    - Anchor reasoning to statutory provisions (Act + Sections + Rules)
    - Add precedent-style reasoning (observational, not fabricated)
    - Emit STRICT ClauseAnalysisResult for aggregation & UI
    """

    # =========================================================
    # Public API
    # =========================================================

    def explain(
        self,
        clause,
        clause_result,
        evidence_pack,
        retrieval_quality: Optional[Dict[str, Any]] = None,
    ) -> ClauseAnalysisResult:

        # -------------------------------------------------
        # 1.5️⃣ Effective confidence (lawyer-facing)
        # -------------------------------------------------
        compliance_conf = clause_result.compliance_confidence or 0.0
        semantic_conf = (
            getattr(clause_result, "semantic_confidence", None)
            or getattr(clause, "semantic_confidence", None)
            or getattr(clause, "confidence", 0.0)
        )

        effective_confidence = min(compliance_conf, semantic_conf)
        if retrieval_quality:
            effective_confidence = min(
                effective_confidence,
                retrieval_quality.get("groundedness_score", 1.0),
            )

        # -------------------------------------------------
        # 1️⃣ Determine stance
        # -------------------------------------------------
        stance = self._determine_stance(
            compliance_confidence=effective_confidence,
            compliance_mode=clause_result.compliance_mode
        )

        # -------------------------------------------------
        # 2️⃣ Determine alignment
        # -------------------------------------------------
        alignment = self._determine_alignment(
            clause_result=clause_result,
            evidence_pack=evidence_pack
        )
        if retrieval_quality and clause_result.compliance_mode != "CONTRADICTION":
            if not retrieval_quality.get("coverage_ok", True) or not retrieval_quality.get("anchor_match", True):
                alignment = "insufficient_evidence"

        # -------------------------------------------------
        # 2b️⃣ Conservative downgrade for unknown intent
        # -------------------------------------------------
        quality_score = effective_confidence

        if clause_result.intent == "unknown":
            alignment = "insufficient_evidence"
            quality_score = min(quality_score, 0.5)

        # -------------------------------------------------
        # 3️⃣ Build explanations (statute-aware)
        # -------------------------------------------------
        plain_summary, legal_explanation, recommended_action = (
            self._build_explanation(
                clause_result=clause_result,
                stance=stance,
                alignment=alignment
            )
        )
        if retrieval_quality:
            grounding_issues = self._grounding_issues(retrieval_quality)
            if grounding_issues:
                legal_explanation += (
                    "\n\nGrounding check: "
                    + " ".join(grounding_issues[:2])
                )

        # -------------------------------------------------
        # 4️⃣ Build statutory references (for UI + lawyers)
        # -------------------------------------------------
        statutory_refs = self._build_statutory_refs(clause_result)

        # -------------------------------------------------
        # 5️⃣ Build STRICT payload
        # -------------------------------------------------
        data = {
            "clause_id": clause.chunk_id,
            "risk_level": clause_result.risk_level,
            "alignment": alignment,

            # Tiered explanations
            "plain_summary": plain_summary,
            "legal_explanation": legal_explanation,

            # Scores
            "quality_score": round(quality_score, 2),
            "compliance_confidence": round(effective_confidence, 2),
            "semantic_confidence": round(float(semantic_conf), 2),
            "groundedness_score": (
                round(float(retrieval_quality.get("groundedness_score", 0.0)), 2)
                if retrieval_quality else None
            ),
            "clause_role": getattr(clause_result, "clause_role", None),

            # Action
            "recommended_action": recommended_action,

            # Citations (statutes + retrieved evidence)
            "citations": statutory_refs + [
                {
                    "source": ev.source,
                    "ref": ev.section_or_clause
                }
                for ev in evidence_pack.evidences
            ],
        }

        return build_model(
            ClauseAnalysisResult,
            data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    def _grounding_issues(self, retrieval_quality: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        if not retrieval_quality.get("coverage_ok", True):
            issues.append("Expected statutory material was not retrieved.")
        if not retrieval_quality.get("anchor_match", True):
            expected = retrieval_quality.get("expected_sections") or []
            if expected:
                issues.append(
                    "Retrieved evidence did not match expected sections: "
                    + ", ".join(expected)
                    + "."
                )
            else:
                issues.append("Retrieved evidence did not match expected statutory anchors.")
        if retrieval_quality.get("noise_penalty", 0.0) > 0.5:
            issues.append("High retrieval noise detected; irrelevant evidence may be present.")
        return issues

    # =========================================================
    # Stance determination
    # =========================================================

    def _determine_stance(
        self,
        compliance_confidence: float,
        compliance_mode: str
    ) -> str:
        if compliance_mode == "CONTRADICTION":
            return "VIOLATION"
        if compliance_confidence >= 0.8:
            return "ASSERTIVE"
        if compliance_confidence >= 0.5:
            return "CAUTIOUS"
        return "WARNING"

    # =========================================================
    # Alignment determination
    # =========================================================

    def _determine_alignment(self, clause_result, evidence_pack) -> str:
        if clause_result.compliance_mode == "CONTRADICTION":
            return "contradiction"
        if not evidence_pack.evidences:
            return "insufficient_evidence"
        if clause_result.compliance_mode == "IMPLICIT":
            return "aligned"
        return "partially_aligned"

    # =========================================================
    # Explanation Builder (tiered + lawyer-safe)
    # =========================================================

    def _build_explanation(
        self,
        clause_result,
        stance: str,
        alignment: str
    ) -> Tuple[str, str, str]:

        intent = clause_result.intent.replace("_", " ")
        statutory_text = self._statutory_text(clause_result)
        precedent = self._precedent_anchor(clause_result.intent)

        # -----------------------------
        # ASSERTIVE
        # -----------------------------
        if stance == "ASSERTIVE":
            return (
                f"This clause complies with RERA requirements relating to {intent}.",
                self._assertive_text(intent, statutory_text, precedent),
                "No action required."
            )

        # -----------------------------
        # CAUTIOUS
        # -----------------------------
        if stance == "CAUTIOUS":
            return (
                f"This clause broadly aligns with RERA provisions on {intent}, "
                f"but could benefit from clearer wording.",
                self._cautious_text(intent, statutory_text, precedent),
                "Review this clause alongside the applicable RERA provisions."
            )

        # -----------------------------
        # WARNING
        # -----------------------------
        if stance == "WARNING":
            return (
                f"This clause may pose legal risk in relation to {intent}.",
                self._warning_text(intent, statutory_text, precedent),
                "Seek clarification or legal review before relying on this clause."
            )

        # -----------------------------
        # VIOLATION
        # -----------------------------
        return (
            "This clause may conflict with mandatory RERA protections.",
            self._violation_text(statutory_text, precedent),
            "Do not rely on this clause; seek immediate legal advice."
        )

    # =========================================================
    # Lawyer-grade templates
    # =========================================================

    def _assertive_text(
        self,
        intent: str,
        statutory: Optional[str],
        precedent: Optional[str]
    ) -> str:
        text = (
            f"The clause addresses {intent} and reflects protections provided under "
            f"the Real Estate (Regulation and Development) Act, 2016."
        )
        if statutory:
            text += f" It preserves statutory rights under {statutory}."
        if precedent:
            text += f"\n\nObserved RERA position: {precedent}"
        return text

    def _cautious_text(
        self,
        intent: str,
        statutory: Optional[str],
        precedent: Optional[str]
    ) -> str:
        text = (
            f"The clause refers to {intent} but relies on statutory incorporation "
            f"rather than explicit contractual wording."
        )
        if statutory:
            text += f" Relevant statutory provisions include {statutory}."
        if precedent:
            text += f"\n\nObserved RERA position: {precedent}"
        return text

    def _warning_text(
        self,
        intent: str,
        statutory: Optional[str],
        precedent: Optional[str]
    ) -> str:
        text = (
            f"The clause relates to {intent}, but its alignment with RERA protections "
            f"is unclear and may affect enforceability."
        )
        if statutory:
            text += f" This may dilute rights conferred under {statutory}."
        if precedent:
            text += f"\n\nObserved RERA position: {precedent}"
        return text

    def _violation_text(
        self,
        statutory: Optional[str],
        precedent: Optional[str]
    ) -> str:
        text = (
            "The clause appears to restrict or waive rights guaranteed under RERA. "
            "Such provisions are generally treated as unenforceable by RERA authorities."
        )
        if statutory:
            text += f" This conflicts with {statutory}."
        if precedent:
            text += f"\n\nObserved RERA position: {precedent}"
        return text

    # =========================================================
    # Statutory anchoring
    # =========================================================

    def _statutory_text(self, clause_result) -> Optional[str]:
        """
        Converts statutory_basis into readable legal text.
        """
        basis = normalize_statutory_basis(
            getattr(clause_result, "statutory_basis", None)
        )
        if not basis:
            return None

        act = basis.get("act", "the RERA Act")
        sections = basis.get("sections", [])
        rules = basis.get("state_rules", [])

        parts = []
        if sections:
            parts.append(f"{', '.join(sections)} of {act}")
        if rules:
            parts.append(f"read with {', '.join(rules)}")

        return " ".join(parts) if parts else None

    def _build_statutory_refs(self, clause_result) -> List[Dict[str, str]]:
        """
        Structured statutory citations for UI / downstream systems.
        """
        refs = []
        basis = normalize_statutory_basis(
            getattr(clause_result, "statutory_basis", None)
        )
        if not basis:
            return refs

        act = basis.get("act", "RERA Act")
        for sec in basis.get("sections", []):
            refs.append({"source": act, "ref": sec})

        for rule in basis.get("state_rules", []):
            refs.append({"source": "State RERA Rules", "ref": rule})

        return refs

    # =========================================================
    # Precedent anchoring (observational)
    # =========================================================

    def _precedent_anchor(self, intent: str) -> Optional[str]:
        """
        Observed RERA authority outcomes (NOT fabricated case law).
        """
        PRECEDENT_MAP = {
            "delay_in_possession": (
                "RERA authorities have consistently held promoters liable for "
                "possession delays where statutory remedies under Section 18 are "
                "not explicitly preserved."
            ),
            "refund_and_withdrawal": (
                "Authorities commonly award refund with interest where withdrawal "
                "rights are restricted contrary to Section 18 of the Act."
            ),
            "unilateral_modification": (
                "Unilateral modification clauses are frequently read down by "
                "RERA authorities as being contrary to Section 14."
            ),
            "jurisdiction": (
                "Clauses excluding RERA authority jurisdiction are routinely "
                "held void in view of Sections 31 and 79 of the Act."
            ),
        }
        return PRECEDENT_MAP.get(intent)
