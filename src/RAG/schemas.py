from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class UserContractChunkSchema(BaseModel):
    clause_id: str
    text: str
    title: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalQuerySchema(BaseModel):
    index: Literal["legal_authority_index", "contract_pattern_index"]
    intent: str
    filters: Dict


class ClauseUnderstandingResultSchema(BaseModel):
    clause_id: str
    intent: str
    obligation_type: Literal["builder", "buyer", "mutual", "unclear"]
    risk_level: Literal["low", "medium", "high"]
    needs_legal_validation: bool
    retrieval_queries: List[RetrievalQuerySchema]
    notes: Optional[List[str]] = None


class EvidenceSchema(BaseModel):
    source: str
    section_or_clause: str
    text: str
    metadata: Dict


class EvidencePackSchema(BaseModel):
    clause_id: str
    intent: str
    evidences: List[EvidenceSchema]


class ExplanationResultSchema(BaseModel):
    clause_id: str
    alignment: Literal[
        "aligned",
        "partially_aligned",
        "conflicting",
        "insufficient_evidence"
    ]
    risk_level: Literal["low", "medium", "high"]
    summary: str
    detailed_explanation: str
    citations: List[Dict]
    quality_score: float = Field(ge=0.0, le=1.0)
    disclaimer: str
