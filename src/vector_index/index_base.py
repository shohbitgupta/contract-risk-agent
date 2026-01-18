from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class IndexDocument:
    """
    Represents a single indexed legal knowledge chunk.

    This is the atomic unit stored in vector indexes and
    retrieved as legal evidence.
    """

    # -------------------------------------------------
    # Core content
    # -------------------------------------------------
    content: str

    # -------------------------------------------------
    # Metadata for retrieval, citation, audit
    # -------------------------------------------------
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    def __post_init__(self):
        if not self.content or len(self.content.strip()) < 50:
            raise ValueError(
                "IndexDocument content is empty or too short"
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("IndexDocument metadata must be a dict")

        # Enforce minimum required metadata
        required_keys = {"source", "chunk_id"}
        missing = required_keys - self.metadata.keys()
        if missing:
            raise ValueError(
                f"IndexDocument metadata missing required keys: {missing}"
            )

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def citation(self) -> str:
        """
        Human-readable citation string.
        Used by LegalExplanationAgent.
        """
        source = self.metadata.get("source", "unknown")
        section = (
            self.metadata.get("section")
            or self.metadata.get("rule")
            or self.metadata.get("clause")
            or self.metadata.get("chunk_id")
        )

        return f"{source} ({section})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize for JSON storage.
        """
        return {
            "content": self.content,
            "metadata": self.metadata
        }