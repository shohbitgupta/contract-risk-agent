from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Base (STRICT)
# -------------------------------------------------------------------

class StrictBaseModel(BaseModel):
    model_config = {
        "extra": "forbid"
    }


# -------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------

class ChunkMetadata(StrictBaseModel):
    doc_type: str
    jurisdiction: str
    state: Optional[str] = None
    source: str
    version: str
    section_or_clause: str
    title: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# User Contract Chunk
# -------------------------------------------------------------------

class UserContractChunk(StrictBaseModel):
    clause_id: str
    text: str
    title: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# -------------------------------------------------------------------
# Clause Understanding Agent Output
# -------------------------------------------------------------------

class ClauseUnderstandingResult(StrictBaseModel):
    """
    Structured output of the Clause Understanding Agent.

    This represents legal interpretation and grounding,
    NOT a final legal conclusion.
    """

    # -----------------------------
    # Clause identity
    # -----------------------------
    clause_id: str

    # -----------------------------
    # Intent & obligation
    # -----------------------------
    intent: str
    obligation_type: str  # promoter | allottee | mutual | unclear

    # -----------------------------
    # Risk & validation
    # -----------------------------
    risk_level: str  # low | medium | high
    needs_legal_validation: bool

    # -----------------------------
    # Retrieval instructions
    # -----------------------------
    retrieval_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vector retrieval instructions derived from intent rules"
    )

    # -----------------------------
    # Compliance interpretation
    # -----------------------------
    compliance_mode: Optional[str] = Field(
        default=None,
        description="IMPLICIT | EXPLICIT | CONTRADICTION | UNKNOWN"
    )

    compliance_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for compliance interpretation"
    )

    # -----------------------------
    # Statutory anchoring (NEW)
    # -----------------------------
    statutory_basis: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Applicable statutory provisions. Example: {"
            "'act': 'RERA Act, 2016', "
            "'sections': ['Section 18(1)'], "
            "'state_rules': ['UP RERA Rules, Rule 16']"
            "}"
        )
    )

    # -----------------------------
    # Diagnostics
    # -----------------------------
    notes: Optional[List[str]] = Field(
        default=None,
        description="Diagnostic notes for ambiguity or edge cases"
    )

    # -----------------------------
    # Semantic confidence
    # -----------------------------
    semantic_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic confidence score for the clause"
    )


    # -----------------------------
    # Clause role
    # -----------------------------
    clause_role: Optional[str] = Field(
        default=None,
        description="Role of the clause in the contract (promoter | allottee | mutual | unclear)"
    )



# -------------------------------------------------------------------
# Evidence
# -------------------------------------------------------------------

class Evidence(StrictBaseModel):
    source: str
    section_or_clause: str
    text: str
    metadata: ChunkMetadata


# -------------------------------------------------------------------
# Evidence Pack
# -------------------------------------------------------------------

class EvidencePack(StrictBaseModel):
    clause_id: str
    clause_text: str
    risk_level: str
    evidences: List[Evidence]
    resolution: Optional[str] = None


# -------------------------------------------------------------------
# Final Explanation Output
# -------------------------------------------------------------------

class ExplanationResult(StrictBaseModel):
    clause_id: str
    alignment: str
    risk_level: str
    summary: str
    detailed_explanation: str
    citations: List[Dict[str, Any]]
    quality_score: float = Field(ge=0.0, le=1.0)
    disclaimer: str
