from RAG.models import ExplanationResult
from tools.logger import setup_logger

from utils.schema_factory import build_model
from utils.schema_drift import log_schema_drift
from configs.schema_config import STRICT_SCHEMA

logger = setup_logger("legal-explanation-agent")


class LegalExplanationAgent:
    """
    Generates legally grounded explanations using:
    - Retrieved evidence
    - Compliance mode
    - Compliance confidence
    """

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def explain(
        self,
        clause,
        clause_result,
        evidence_pack
    ) -> ExplanationResult:

        # 1️⃣ Determine explanation stance
        stance = self._determine_stance(
            compliance_confidence=clause_result.compliance_confidence,
            compliance_mode=clause_result.compliance_mode
        )

        # 2️⃣ Determine alignment label
        alignment = self._determine_alignment(
            clause_result=clause_result,
            evidence_pack=evidence_pack
        )

        # 3️⃣ Build explanation text
        summary, detailed = self._build_explanation(
            clause=clause,
            clause_result=clause_result,
            evidence_pack=evidence_pack,
            stance=stance
        )

        # -------------------------------------------------
        # Build schema-safe payload
        # -------------------------------------------------

        data = {
            "clause_id": getattr(clause, "clause_id", getattr(clause, "chunk_id")),
            "risk_level": clause_result.risk_level,
            "alignment": alignment,
            "summary": summary,
            "detailed_explanation": detailed,
            "citations": [
                {
                    "source": ev.source,
                    "section_or_clause": ev.section_or_clause
                }
                for ev in evidence_pack.evidences
            ],
            "quality_score": clause_result.compliance_confidence,
            "disclaimer": (
                "This explanation is generated for informational purposes only and does not "
                "constitute legal advice. Independent legal review is recommended."
            ),
        }

        return build_model(
            ExplanationResult,
            data,
            strict=STRICT_SCHEMA,
            log_fn=log_schema_drift
        )

    # -------------------------------------------------
    # Stance determination
    # -------------------------------------------------

    def _determine_stance(self, compliance_confidence: float, compliance_mode: str) -> str:
        if compliance_mode == "CONTRADICTION":
            return "VIOLATION"

        if compliance_confidence >= 0.8:
            return "ASSERTIVE"

        if compliance_confidence >= 0.5:
            return "CAUTIOUS"

        return "WARNING"

    # -------------------------------------------------
    # Alignment determination
    # -------------------------------------------------

    def _determine_alignment(self, clause_result, evidence_pack) -> str:
        if clause_result.compliance_mode == "CONTRADICTION":
            return "conflicting"

        if not evidence_pack.evidences:
            return "insufficient_evidence"

        if clause_result.compliance_mode == "IMPLICIT":
            return "aligned"

        return "partially_aligned"

    # -------------------------------------------------
    # Explanation Builder
    # -------------------------------------------------

    def _build_explanation(
        self,
        clause,
        clause_result,
        evidence_pack,
        stance: str
    ) -> tuple[str, str]:

        intent = clause_result.intent.replace("_", " ").title()

        if stance == "ASSERTIVE":
            return self._assertive_template(clause, clause_result, evidence_pack, intent)

        if stance == "CAUTIOUS":
            return self._cautious_template(clause, clause_result, evidence_pack, intent)

        if stance == "WARNING":
            return self._warning_template(clause, clause_result, evidence_pack, intent)

        return self._violation_template(clause, clause_result, evidence_pack, intent)

    # -------------------------------------------------
    # Templates
    # -------------------------------------------------

    def _assertive_template(self, clause, clause_result, evidence_pack, intent):
        summary = (
            f"This clause complies with RERA requirements relating to {intent.lower()}."
        )

        detailed = (
            f"The reviewed clause addresses {intent.lower()} and explicitly or implicitly "
            f"incorporates the protections provided under the Real Estate (Regulation and "
            f"Development) Act, 2016. The clause follows the standard structure prescribed "
            f"under RERA and does not dilute the statutory rights of the allottee.\n\n"
            f"No deviation from applicable RERA provisions has been identified."
        )

        return summary, detailed

    def _cautious_template(self, clause, clause_result, evidence_pack, intent):
        summary = (
            f"This clause appears to align with RERA provisions on {intent.lower()}, "
            f"subject to interpretation."
        )

        detailed = (
            f"The clause addresses {intent.lower()} and refers to obligations under the "
            f"RERA Act or applicable rules. While no direct contradiction with RERA has "
            f"been identified, the clause relies on statutory incorporation rather than "
            f"explicit articulation of rights.\n\n"
            f"It is advisable to review this clause in conjunction with the applicable "
            f"RERA provisions to ensure full clarity."
        )

        return summary, detailed

    def _warning_template(self, clause, clause_result, evidence_pack, intent):
        summary = (
            f"This clause may pose legal risk in relation to {intent.lower()}."
        )

        detailed = (
            f"The clause relates to {intent.lower()}, but its alignment with RERA "
            f"protections is unclear or incomplete. The language used may result in "
            f"ambiguity regarding the allottee’s statutory rights.\n\n"
            f"Independent legal review is recommended before relying on this clause."
        )

        return summary, detailed

    def _violation_template(self, clause, clause_result, evidence_pack, intent):
        summary = (
            f"This clause is not compliant with RERA and may be legally unenforceable."
        )

        detailed = (
            f"The clause attempts to restrict or waive rights guaranteed to the allottee "
            f"under the RERA framework. Such provisions are not permitted under the Act "
            f"and are likely to be struck down by the RERA Authority or Adjudicating Officer.\n\n"
            f"The allottee should not rely on this clause as it conflicts with statutory law."
        )

        return summary, detailed
