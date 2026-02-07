# RAG/models/contract_analysis.py

from typing import List, Dict, Optional
from pydantic import Field
from RAG.models import StrictBaseModel


class ContractRiskDistribution(StrictBaseModel):
    aligned: int
    partially_aligned: int
    insufficient_evidence: int
    contradiction: int


class ContractSummary(StrictBaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    legal_confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    distribution: ContractRiskDistribution


class KeyIssue(StrictBaseModel):
    clause_id: str
    risk_level: str
    issue: str
    recommended_action: str
    quality_score: float = Field(ge=0.0, le=1.0)


class ClauseAnalysisResult(StrictBaseModel):
    clause_id: str
    # (lawyer-grade reference)
    normalized_reference: Optional[str] = None   # "Clause 7.3"
    heading: Optional[str] = None                # "Possession Timeline"
    statutory_refs: List[str] = []               # ["RERA Act, 2016 – Section 18"]

    risk_level: str
    alignment: str

    plain_summary: str
    legal_explanation: str

    quality_score: float = Field(ge=0.0, le=1.0)
    compliance_confidence: float = Field(ge=0.0, le=1.0)

    citations: List[Dict[str, str]] = []
    recommended_action: Optional[str] = None
    issue_reason: Optional[str] = None



class ContractAnalysisResult(StrictBaseModel):
    contract_summary: ContractSummary
    top_issues: List[KeyIssue]
    clauses: List[ClauseAnalysisResult]

    @property
    def contract_score(self) -> float:
        return self.contract_summary.overall_score

    @property
    def risk_grade(self) -> str:
        return self.contract_summary.risk_level
