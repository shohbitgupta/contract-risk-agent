from typing import Dict, Any
from RAG.models import ChunkMetadata


def normalize_chunk_metadata(
    raw: Dict[str, Any],
    *,
    index_name: str,
    state: str | None
) -> Dict[str, Any]:
    """
    Normalize raw vector-index metadata into ChunkMetadata-compatible dict.
    """

    return {
        "doc_type": raw.get("doc_type") or _infer_doc_type(index_name),
        "jurisdiction": raw.get("jurisdiction", "india"),
        "state": raw.get("state", state),
        "source": raw.get("source", index_name),
        "version": raw.get("version", "unknown"),
        "section_or_clause": (
            raw.get("section")
            or raw.get("rule")
            or raw.get("clause")
            or raw.get("chunk_id")
            or "UNKNOWN"
        ),
        "title": raw.get("title"),
        "extra": raw,  # preserve everything else for audit
    }

def _infer_doc_type(index_name: str) -> str:
    mapping = {
        "model_bba": "model_agreement",
        "rera_act": "rera_act",
        "rera_rules": "state_rule",
        "circulars": "notification",
        "case_law": "case_law",
    }
    return mapping.get(index_name, "unknown")
