import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ChunkType(Enum):
    """
    Classifies the structural role of the chunk.

    Example:
        >>> ChunkType.CLAUSE.value
        'clause'
    """
    CLAUSE = "clause"  # Standard numbered clause (1.1, 12.4)
    DEFINITION = "definition"  # Specific definition entries (e.g., "Act means...")
    SCHEDULE = "schedule"  # Annexures, Schedules, Tables
    SECTION_HEADER = "header"  # High-level headers (A., B., PART I)
    NOTE = "note"  # Disclaimers, Notes, Important blocks
    PARAGRAPH = "paragraph"  # Fallback text blocks


class RiskLevel(Enum):
    """
    Risk severity based on semantic tagging.

    Example:
        >>> RiskLevel.HIGH.value
        'high'
    """
    HIGH = "high"  # Penalties, Termination, Liability
    MEDIUM = "medium"  # Financials, Dates
    LOW = "low"  # Descriptive, Boilerplate
    UNKNOWN = "unknown"


from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ContractChunk:
    """
    Domain object representing a single distinct unit of the contract.
    """

    # Core identity
    chunk_id: str                    # "(iii)", "1.2", "SCHEDULE-A"
    text: str
    chunk_type: ChunkType

    # Structural metadata
    title: Optional[str] = None      # "Possession Timeline"
    parent_section: Optional[str] = None   # "Clause 7"
    normalized_reference: Optional[str] = None  # "Clause 7.3"
    page_number: Optional[int] = None
    confidence: float = 0.0

    # Semantic enrichment
    tags: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.UNKNOWN

    # Legal enrichment
    statutory_refs: List[str] = field(default_factory=list)
    intent: Optional[str] = None  # possession_delay, refund_delay, etc.

    # Semantic enrichment
    semantic_confidence: float = 0.0

    def to_dict(self):
        return {
            "id": self.chunk_id,
            "normalized_reference": self.normalized_reference,
            "parent_section": self.parent_section,
            "title": self.title,
            "type": self.chunk_type.value,
            "text": self.text,
            "tags": self.tags,
            "risk": self.risk_level.value,
            "confidence": self.confidence,
            "statutory_refs": self.statutory_refs,
            "intent": self.intent,
            "semantic_confidence": self.semantic_confidence,
        }


import re
from typing import List, Optional


class UserContractChunker:
    """
    Optimized Chunker for Indian Real Estate Contracts (BBA / AFS / RERA).
    Produces structurally correct ContractChunk objects.

    Example:
        >>> chunker = UserContractChunker()
        >>> chunks = chunker.chunk("1. DEFINITIONS\\n2. POSSESSION ...")
        >>> len(chunks) > 0
        True
    """

    # --------------------------------------------------
    # Clause detection patterns (ordered by priority)
    # --------------------------------------------------

    CLAUSE_PATTERNS = [
        r"(^(SCHEDULE|ANNEXURE)\s*[-A-Z0-9]+)",
        r"((ARTICLE|SECTION|CLAUSE)\s+([0-9]+|[IVX]+)(\.[0-9]+)?)",
        r"(^([A-Z]|\d+)\.\s)",
        r"(^\d+(\.\d+)+(\([a-z0-9]+\))?)",
        r"(^\([a-zA-Z0-9]+\)\s)",
        r"(NOW THIS DEED WITNESSETH|IN WITNESS WHEREOF|DEFINITIONS)"
    ]

    # --------------------------------------------------
    # RERA semantic anchors (unnumbered but critical)
    # --------------------------------------------------

    RERA_SEMANTIC_HEADERS = [
        "FORCE MAJEURE",
        "DELAY IN POSSESSION",
        "HANDING OVER OF POSSESSION",
        "DATE OF POSSESSION",
        "RATE OF INTEREST",
        "DEFECT LIABILITY",
        "FORMATION OF ASSOCIATION",
        "COMMON AREAS",
        "JURISDICTION"
    ]

    TITLE_BLOCKLIST = {
        "whereas",
        "now this deed",
        "in witness whereof"
    }

    # --------------------------------------------------

    def chunk(self, text: str) -> List[ContractChunk]:
        """
        Split raw contract text into structurally labeled chunks.

        Returns:
            List of ContractChunk objects.
        """
        normalized = self._normalize(text)
        raw_clauses = self._split_into_raw_clauses(normalized)

        chunks: List[ContractChunk] = []

        for cid, clause_text in raw_clauses:
            clean_text = clause_text.strip()
            chunk_type = self._detect_chunk_type(cid)
            title = self._extract_title(clean_text)

            # Sub-chunk definitions
            if (
                len(clean_text) > 1500
                and title
                and "definition" in title.lower()
            ):
                chunks.extend(self._sub_chunk_definitions(cid, clean_text))
                continue

            # Sub-chunk large schedules
            if chunk_type == ChunkType.SCHEDULE and len(clean_text) > 1200:
                chunks.extend(self._sub_chunk_schedule(cid, clean_text))
                continue

            chunks.append(self._build_chunk(cid, clean_text, chunk_type, title))

        return chunks

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    def _normalize(self, text: str) -> str:
        """
        Normalize text while preserving legal structure.

        This cleans line endings and standardizes schedule headers.
        """
        text = text.replace("\r", "")
        text = re.sub(
            r"Schedule\s*-\s*([A-Z])",
            r"Schedule \1",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"(\s)(\([a-z]\)\s)",
            r"\n\2",
            text
        )
        return text.strip()

    # --------------------------------------------------
    # Clause splitting
    # --------------------------------------------------

    def _split_into_raw_clauses(self, text: str):
        """
        Detect clause boundaries using regex patterns and semantic headers.
        """
        combined = "|".join(self.CLAUSE_PATTERNS)
        regex = re.compile(combined, re.MULTILINE | re.IGNORECASE)

        matches = list(regex.finditer(text))

        # Inject RERA semantic headers
        for header in self.RERA_SEMANTIC_HEADERS:
            for match in re.finditer(
                rf"^{header}\b",
                text,
                re.MULTILINE | re.IGNORECASE
            ):
                matches.append(match)

        if len(matches) < 3:
            return self._fallback_split(text)

        matches = sorted(matches, key=lambda m: m.start())
        return self._split_by_matches(text, matches)

    def _split_by_matches(self, text: str, matches):
        """
        Split text into clause segments based on regex match spans.
        """
        clauses = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            clauses.append((match.group().strip(), text[start:end]))
        return clauses

    # --------------------------------------------------
    # Chunk creation
    # --------------------------------------------------

    def _build_chunk(
        self,
        cid: str,
        text: str,
        chunk_type: ChunkType,
        title: Optional[str]
    ) -> ContractChunk:
        """
        Construct a ContractChunk with metadata fields populated.
        """
        semantic_confidence = self._compute_semantic_confidence(cid, text, chunk_type, title)

        return ContractChunk(
            chunk_id=cid,
            text=text,
            chunk_type=chunk_type,
            title=title,
            page_number=None,
            confidence=self._confidence_for(cid),
            tags=[],
            semantic_confidence=semantic_confidence,
            risk_level=RiskLevel.UNKNOWN
        )

    def _confidence_for(self, cid: str) -> float:
        """
        Heuristic confidence score for chunk ids.
        """
        return 0.9 if cid and cid[0].isalnum() else 0.6

    def _detect_chunk_type(self, cid: str) -> ChunkType:
        """
        Infer chunk type from the clause id format.
        """
        cid_upper = cid.upper()
        if "SCHEDULE" in cid_upper or "ANNEXURE" in cid_upper:
            return ChunkType.SCHEDULE
        if cid.strip().startswith("(") and cid.strip().endswith(")"):
            return ChunkType.DEFINITION
        if re.match(r"^[A-Z]\.$", cid.strip()):
            return ChunkType.SECTION_HEADER
        return ChunkType.CLAUSE

    # --------------------------------------------------
    # Title extraction
    # --------------------------------------------------

    def _extract_title(self, text: str) -> Optional[str]:
        """
        Extract a short title/header from the clause text.
        """
        lines = text.split("\n")
        if not lines:
            return None

        first = lines[0].strip()
        if len(first) < 5 and len(lines) > 1:
            first = lines[1].strip()

        clean = re.sub(
            r"^([A-Z0-9]+\.?\s+|\([a-z]\)\s+)",
            "",
            first
        )

        clean_lower = clean.lower()
        if any(b in clean_lower for b in self.TITLE_BLOCKLIST):
            return None

        return clean if len(clean) < 100 else None

    # --------------------------------------------------
    # Sub-chunking
    # --------------------------------------------------

    def _sub_chunk_definitions(self, parent_id: str, text: str) -> List[ContractChunk]:
        """
        Split long definition blocks into individual definition chunks.
        """
        pattern = r"(^\([a-z0-9]+\)\s)"
        regex = re.compile(pattern, re.MULTILINE)
        matches = list(regex.finditer(text))

        if len(matches) < 2:
            return [self._build_chunk(parent_id, text, ChunkType.DEFINITION, None)]

        chunks = []

        if matches[0].start() > 0:
            chunks.append(
                self._build_chunk(
                    f"{parent_id}_PREAMBLE",
                    text[:matches[0].start()],
                    ChunkType.CLAUSE,
                    None
                )
            )

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            cid = f"{parent_id}{match.group().strip()}"
            chunks.append(
                self._build_chunk(
                    cid,
                    text[start:end],
                    ChunkType.DEFINITION,
                    None
                )
            )

        return chunks

    def _sub_chunk_schedule(self, cid: str, text: str) -> List[ContractChunk]:
        """
        Split long schedules into smaller parts.
        """
        parts = re.split(r"\n\s*(\d+\.|\([a-z]\)|-)\s+", text)

        if len(parts) < 3:
            return [self._build_chunk(cid, text, ChunkType.SCHEDULE, None)]

        chunks = []
        for i, part in enumerate(parts):
            if part.strip():
                chunks.append(
                    self._build_chunk(
                        f"{cid}_PART_{i}",
                        part.strip(),
                        ChunkType.SCHEDULE,
                        None
                    )
                )

        return chunks

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    def _fallback_split(self, text: str):
        """
        Fallback paragraph split when clause patterns are not detected.
        """
        paragraphs = text.split("\n\n")
        clauses = []

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            cid = (
                f"NOTE_{idx}"
                if re.match(r"^(Note|Important|Disclaimer):", para, re.IGNORECASE)
                else f"PARA_{idx}"
            )
            clauses.append((cid, para))

        return clauses

    def _compute_semantic_confidence(
        self,
        cid: str,
        text: str,
        chunk_type: ChunkType,
        title: Optional[str]
    ) -> float:
        """
        Compute semantic confidence score for a chunk.
        """
        score = 0.0

    # 1️⃣ Structural validity
        if cid and any(char.isdigit() for char in cid):
         score += 0.2

    # 3️⃣ Title clarity
        if title and len(title) > 5:
         score += 0.15

    # 4️⃣ Text sufficiency
        if len(text) > 200:
          score += 0.1
        if len(text) > 500:
          score += 0.1

        return round(min(score, 0.7), 2)
