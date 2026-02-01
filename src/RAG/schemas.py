from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class UserContractChunkSchema(BaseModel):
    """
    Pydantic schema for serialized user contract chunks.

    Example:
        >>> UserContractChunkSchema(clause_id="1.1", text="...", confidence=0.8)
    """
    clause_id: str
    text: str
    title: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalQuerySchema(BaseModel):
    """
    Schema for retrieval query instructions.

    Example:
        >>> RetrievalQuerySchema(index="legal_authority_index", intent="delay", filters={})
    """
    index: Literal["legal_authority_index", "contract_pattern_index"]
    intent: str
    filters: Dict


class ClauseUnderstandingResultSchema(BaseModel):
    """
    Schema for clause understanding output.

    Example:
        >>> ClauseUnderstandingResultSchema(clause_id="1.1", intent="possession_delay",
        ...                                 obligation_type="builder", risk_level="medium",
        ...                                 needs_legal_validation=True, retrieval_queries=[])
    """
    clause_id: str
    intent: str
    obligation_type: Literal["builder", "buyer", "mutual", "unclear"]
    risk_level: Literal["low", "medium", "high"]
    needs_legal_validation: bool
    retrieval_queries: List[RetrievalQuerySchema]
    notes: Optional[List[str]] = None


class EvidenceSchema(BaseModel):
    """
    Schema for a single evidence snippet.

    Example:
        >>> EvidenceSchema(source="RERA Act", section_or_clause="Section 18",
        ...                text="...", metadata={})
    """
    source: str
    section_or_clause: str
    text: str
    metadata: Dict


class EvidencePackSchema(BaseModel):
    """
    Schema for evidence pack returned by retrieval.

    Example:
        >>> EvidencePackSchema(clause_id="1.1", intent="possession_delay", evidences=[])
    """
    clause_id: str
    intent: str
    evidences: List[EvidenceSchema]


class ExplanationResultSchema(BaseModel):
    """
    Schema for final explanation output.

    Example:
        >>> ExplanationResultSchema(clause_id="1.1", alignment="aligned",
        ...                         risk_level="low", summary="...", detailed_explanation="...",
        ...                         citations=[], quality_score=0.9, disclaimer="...")
    """
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
