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
    clause_id: str
    intent: str
    obligation_type: str
    risk_level: str
    needs_legal_validation: bool

    retrieval_queries: List[Dict[str, Any]]

    compliance_mode: Optional[str] = None

    notes: Optional[List[str]] = None
    compliance_confidence: Optional[float] = None


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
