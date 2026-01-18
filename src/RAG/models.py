from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class ChunkMetadata:
    doc_type: str                 # RERA_ACT | state_rule | notification | model_agreement
    jurisdiction: str             # india
    state: Optional[str]          # maharashtra | karnataka | None
    source: str                   # MahaRERA, Central Act, etc.
    version: str                  # 2016 / 2023-01
    section_or_clause: str         # Section 18 / Clause 12.4
    title: Optional[str] = None
    extra: Optional[Dict] = None


@dataclass
class UserContractChunk:
    """
    Ephemeral chunk representing a single clause/obligation
    from a user-uploaded contract.
    """

    clause_id: str                 # e.g. "1.1", "12.4", "UNNUM_3"
    text: str                      # full clause text
    title: Optional[str] = None    # optional heading
    confidence: float = 0.0


@dataclass
class ExplanationResult:
    clause_id: str
    alignment: str                 # aligned | partially_aligned | conflicting | insufficient_evidence
    risk_level: str
    summary: str
    detailed_explanation: str
    citations: List[Dict]
    quality_score: float
    disclaimer: str


@dataclass
class ClauseUnderstandingResult:
    """
    Output of the Clause Understanding Agent.
    This is a structured interpretation of a single contract clause,
    NOT a legal conclusion.
    """

    clause_id: str
    intent: str                         # e.g. delay_possession, payment_terms
    obligation_type: str               # builder | buyer | mutual | unclear
    risk_level: str                    # low | medium | high
    needs_legal_validation: bool

    # Structured retrieval instructions for the Retrieval Orchestrator
    retrieval_queries: List[Dict]

    # Optional diagnostic notes (edge cases, ambiguity, etc.)
    notes: Optional[List[str]] = None


@dataclass
class Evidence:
    """
    A single piece of retrieved legal or contractual evidence.
    """

    source: str                        # e.g. RERA Act 2016, MahaRERA Model Agreement
    section_or_clause: str             # Section 18, Clause 10.1, etc.
    text: str                          # Exact retrieved chunk text
    metadata: Dict                     # Full metadata (state, doc_type, version)


@dataclass
class EvidencePack:
    """
    Collection of evidence retrieved for a single user clause.
    """

    clause_id: str
    intent: str
    evidences: List[Evidence]