"""
Statute section-level parsing + indexing.

Goal: one FAISS document = one statutory section.
This makes anchor matching and citations reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vector_index.embedding import EmbeddingGenerator
from vector_index.faiss_index import FAISSVectorIndex
from vector_index.index_base import IndexDocument


@dataclass(frozen=True)
class ParsedSection:
    section_id: str  # e.g. "18", "18A"
    section_number: Optional[int]  # 18
    section_label: str  # e.g. "Section 18"
    title: Optional[str]  # e.g. "Return of amount and compensation"
    content: str  # full section text including heading


class StatuteSectionIndexer:
    """
    Builds statute indexes with strict section-level chunking.

    Output format matches runtime retrieval stack:
    - `*.faiss` + `*.meta.json`
    - Each document is an `IndexDocument` with required metadata:
      `source`, `chunk_id` (plus rich statute fields)
    """

    # Strategy A (some docs): headings like "Section 18. Title"
    _SECTION_HEADING_RE = re.compile(
        r"(?im)^\s*(section)\s+(\d+[a-zA-Z]*)\b(?:\s*[\.\-–—:]\s*(.*?))?\s*$"
    )

    # Strategy B (RERA Act PDF extraction): numbered sections like "18. (1) ..."
    _NUMERIC_SECTION_RE = re.compile(r"(?m)^\s*(\d{1,4})\.\s+")

    def __init__(
        self,
        *,
        act_name: str,
        doc_type: str = "rera_act",
        state: str = "central",
        embedding_model: str = "all-MiniLM-L6-v2",
        jurisdiction: str = "india",
        version: str = "unknown",
    ):
        self.act_name = act_name
        self.doc_type = doc_type
        self.state = state
        self.jurisdiction = jurisdiction
        self.version = version
        self.embedder = EmbeddingGenerator(model_name=embedding_model)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def parse_sections(self, full_text: str) -> List[ParsedSection]:
        text = self._normalize_for_parsing(full_text)

        # Try Strategy A first
        heading_matches = list(self._SECTION_HEADING_RE.finditer(text))
        if heading_matches:
            return self._parse_from_section_headings(text, heading_matches)

        # Fallback Strategy B (common for bare acts extracted from PDF)
        num_matches = list(self._NUMERIC_SECTION_RE.finditer(text))
        if num_matches:
            return self._parse_from_numeric_sections(text, num_matches)

        raise ValueError(
            "No statutory sections detected. Expected either:\n"
            "- 'Section 18 ...' headings at line starts, or\n"
            "- numbered sections like '18. (1) ...' at line starts."
        )

    def _parse_from_section_headings(
        self, text: str, matches: List[re.Match[str]]
    ) -> List[ParsedSection]:
        sections: List[ParsedSection] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()

            section_id = m.group(2).upper()  # "18A"
            section_number = self._extract_int_prefix(section_id)
            section_label = f"Section {section_id}"
            title = (m.group(3) or "").strip() or None

            if not block.lower().startswith("section"):
                block = f"{section_label}\n{block}"

            sections.append(
                ParsedSection(
                    section_id=section_id,
                    section_number=section_number,
                    section_label=section_label,
                    title=title,
                    content=block,
                )
            )
        return sections

    def _parse_from_numeric_sections(
        self, text: str, matches: List[re.Match[str]]
    ) -> List[ParsedSection]:
        sections: List[ParsedSection] = []

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()

            sec_num_str = m.group(1)  # "18"
            sec_num = int(sec_num_str)
            section_id = sec_num_str
            section_label = f"Section {sec_num_str}"

            title = self._infer_title_from_context(text, start)

            # Rewrite leading "18." -> "Section 18." for anchor-match + clean citations
            block = re.sub(
                rf"(?m)^\s*{re.escape(sec_num_str)}\.\s+",
                f"{section_label}. ",
                block,
                count=1,
            )

            # Ensure we still have an explicit "Section N" at the top
            if not block.lower().lstrip().startswith("section "):
                block = f"{section_label}.\n{block}"

            sections.append(
                ParsedSection(
                    section_id=section_id,
                    section_number=sec_num,
                    section_label=section_label,
                    title=title,
                    content=block,
                )
            )

        return sections

    def build_index(
        self,
        *,
        full_text: str,
        output_dir: Path,
        index_name: str,
        source: Optional[str] = None,
        delete_existing: bool = False,
    ) -> Tuple[Path, int]:
        """
        Build a FAISS index file at:
          `<output_dir>/<index_name>.faiss` (+ sidecar `<index_name>.meta.json`)
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / f"{index_name}.faiss"

        if delete_existing:
            self._delete_if_exists(index_path)
            self._delete_if_exists(index_path.with_suffix(".meta.json"))

        parsed = self.parse_sections(full_text)
        documents: List[IndexDocument] = []

        src = source or self.act_name

        for sec in parsed:
            chunk_id = f"{index_name}::section_{sec.section_id}"
            meta: Dict[str, Any] = {
                # Required by IndexDocument
                "source": src,
                "chunk_id": chunk_id,
                # Retrieval normalization / UI
                "index_name": index_name,
                "doc_type": self.doc_type,
                "jurisdiction": self.jurisdiction,
                "state": self.state,
                "version": self.version,
                # Statute-specific fields for anchor matching + audit
                "act": self.act_name,
                "section": sec.section_label,
                "section_id": sec.section_id,
                "section_number": sec.section_number,
                "title": sec.title,
            }

            documents.append(IndexDocument(content=sec.content, metadata=meta))

        embeddings_list = self.embedder.embed([d.content for d in documents])
        embeddings = np.asarray(embeddings_list, dtype="float32")

        index = FAISSVectorIndex(index_path=index_path, dim=embeddings.shape[1])
        index.add(embeddings=embeddings, documents=documents)
        index.persist()

        return index_path, len(documents)

    # -----------------------------------------------------
    # Internals
    # -----------------------------------------------------

    def _normalize_for_parsing(self, text: str) -> str:
        """
        Minimal normalization that preserves headings at line starts.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # PDF form-feed → treat as page break
        text = text.replace("\x0c", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_int_prefix(self, s: str) -> Optional[int]:
        m = re.match(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def _infer_title_from_context(self, text: str, section_start: int) -> Optional[str]:
        """
        Heuristic: many extracted bare-acts include a short title line before the numbered section.
        Example:
          "Definitions."
          "2. In this Act, ..."
        """
        window = text[max(0, section_start - 500):section_start]
        lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
        if not lines:
            return None

        # Look at last 1-2 non-empty lines before section start
        candidates = lines[-2:]
        for cand in reversed(candidates):
            if len(cand) > 140:
                continue
            if re.match(r"^\d+\.\s", cand):
                continue
            if cand.lower().startswith(("chapter", "part", "the gazette", "registered")):
                continue
            if cand.endswith(".") and not cand.lower().startswith("section"):
                # Strip trailing period for title
                return cand[:-1].strip() or None
        return None

    def _delete_if_exists(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            # Non-fatal; indexing will overwrite if possible
            pass
