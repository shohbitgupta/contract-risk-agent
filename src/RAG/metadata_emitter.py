from RAG.models import ChunkMetadata

def get_chunk_metadata(
    *,
    doc_type: str,
    jurisdiction: str = "india",
    state: str | None,
    source: str,
    version: str,
    section_or_clause: str,
    title: str | None = None,
    extra: dict | None = None
) -> dict:
    """
    Emits normalized metadata dictionary for each chunk.
    """

    metadata = ChunkMetadata(
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        state=state,
        source=source,
        version=version,
        section_or_clause=section_or_clause,
        title=title,
        extra=extra
    )

    return metadata.__dict__