from pathlib import Path
from typing import Dict

from vector_index.faiss_index import FAISSVectorIndex


class IndexRegistry:
    """
    Runtime registry for state-specific legal vector indexes.

    Responsibilities:
    - Validate index availability
    - Load FAISS indexes from disk
    - Cache indexes in memory
    - Serve indexes to RetrievalOrchestrator
    """

    def __init__(self, base_dir: Path, embedding_dim: int):
        self.base_dir = base_dir
        self.embedding_dim = embedding_dim

        # Cache: state -> {index_name -> FAISSVectorIndex}
        self._cache: Dict[str, Dict[str, FAISSVectorIndex]] = {}

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def validate_state(self, state: str):
        """
        Fail fast if vector indexes for a state are missing.
        """
        state_dir = self.base_dir / state
        if not state_dir.exists() or not state_dir.is_dir():
            raise RuntimeError(
                f"Vector indexes for state '{state}' not found at {state_dir}. "
                "Run build_up_rera_indexes.py first."
            )

        faiss_files = list(state_dir.glob("*.faiss"))
        if not faiss_files:
            raise RuntimeError(
                f"No FAISS index files found for state '{state}'. "
                "Index directory is empty."
            )

    def get_indexes(self, state: str) -> Dict[str, FAISSVectorIndex]:
        """
        Returns all vector indexes for a given state.
        Uses in-memory cache after first load.
        """
        if state in self._cache:
            return self._cache[state]

        self.validate_state(state)

        state_dir = self.base_dir / state
        indexes: Dict[str, FAISSVectorIndex] = {}

        for faiss_path in state_dir.glob("*.faiss"):
            index_name = faiss_path.stem  # e.g. rera_act, model_bba

            index = FAISSVectorIndex.load(
                index_path=faiss_path,
                dim=self.embedding_dim
            )

            indexes[index_name] = index

        if not indexes:
            raise RuntimeError(
                f"Failed to load any vector indexes for state '{state}'."
            )

        self._cache[state] = indexes
        return indexes

    # -------------------------------------------------
    # Optional helpers
    # -------------------------------------------------

    def list_states(self):
        """
        Returns all states for which indexes exist.
        """
        if not self.base_dir.exists():
            return []

        return [
            p.name for p in self.base_dir.iterdir()
            if p.is_dir()
        ]

    def clear_cache(self):
        """
        Clears in-memory cache (useful for tests).
        """
        self._cache.clear()
