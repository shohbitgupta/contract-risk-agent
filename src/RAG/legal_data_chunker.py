import re
from typing import List, Tuple

from RAG.metadata_emitter import get_chunk_metadata

class LegalChunker:
    """
    Chunker for legal documents based on document type.

    Example:
        >>> chunker = LegalChunker()
        >>> chunks = chunker.chunk(text="Section 1 ...", doc_type="rera_act", source="rera_act", version="2016")
    """

    # -----------------------------
    # PUBLIC API
    # -----------------------------

    def chunk(
        self,
        *,
        text: str,
        doc_type: str,
        source: str,
        version: str,
        state: str | None = None
    ) -> List[Tuple[str, dict]]:
        """
        Returns list of (chunk_text, metadata_dict)

        Example:
            >>> chunker.chunk(text="Clause 1 ...", doc_type="model_agreement", source="bba", version="2024")
            [('Clause 1 ...', {...})]
        """

        if doc_type in {"rera_act", "state_rule"}:
            return self._chunk_by_section(
                text, doc_type, source, version, state
            )

        if doc_type == "notification":
            return self._chunk_notification(
                text, source, version, state
            )

        if doc_type == "model_agreement":
            return self._chunk_by_clause(
                text, source, version, state
            )

        raise ValueError(f"Unsupported doc_type: {doc_type}")

    # -----------------------------
    # INTERNAL METHODS
    # -----------------------------

    def _chunk_by_section(
        self, text, doc_type, source, version, state
    ):
        """
        Chunk Acts / Rules by Section or Rule number.
        """

        pattern = r"(Section\s+\d+[A-Za-z]*|Rule\s+\d+[A-Za-z]*)"
        splits = re.split(pattern, text)

        chunks = []
        for i in range(1, len(splits), 2):
            section_id = splits[i].strip()
            content = splits[i + 1].strip()

            metadata = get_chunk_metadata(
                doc_type=doc_type,
                state=state,
                source=source,
                version=version,
                section_or_clause=section_id,
                title=None
            )

            chunks.append((content, metadata))

        return chunks

    def _chunk_notification(
        self, text, source, version, state
    ):
        """
        Notifications are atomic — never split.
        """

        metadata = get_chunk_metadata(
            doc_type="notification",
            state=state,
            source=source,
            version=version,
            section_or_clause="full_document"
        )

        return [(text.strip(), metadata)]

    def _chunk_by_clause(
        self, text, source, version, state
    ):
        """
        Chunk model agreements by clause number.
        """

        pattern = r"(Clause\s+\d+(\.\d+)*)"
        splits = re.split(pattern, text)

        chunks = []
        for i in range(1, len(splits), 3):
            clause_id = splits[i].strip()
            content = splits[i + 2].strip()

            metadata = get_chunk_metadata(
                doc_type="model_agreement",
                state=state,
                source=source,
                version=version,
                section_or_clause=clause_id
            )

            chunks.append((content, metadata))

        return chunks
