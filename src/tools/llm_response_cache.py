import json
import hashlib
from pathlib import Path
from typing import Optional, Dict


class LLMResponseCache:
    """
    Deterministic cache for LLM responses.

    Example:
        >>> cache = LLMResponseCache(Path("data/llm_cache"))
        >>> key = cache.build_cache_key("text", "intent", "type", evidence_pack)
        >>> cache.set(key, {"alignment": "aligned"})
    """

    def __init__(self, cache_dir: Path):
        """
        Create the cache directory if it does not exist.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def get(self, cache_key: str) -> Optional[Dict]:
        """
        Load a cached response by key.
        """
        path = self._path_for_key(cache_key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, cache_key: str, value: Dict):
        """
        Store a response in the cache.
        """
        path = self._path_for_key(cache_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def build_cache_key(
        self,
        clause_text: str,
        intent: str,
        obligation_type: str,
        evidence_pack
    ) -> str:
        """
        Stable fingerprint for a clause explanation request.

        Example:
            >>> cache.build_cache_key("text", "intent", "type", evidence_pack)
            '...sha256...'
        """

        evidence_fingerprint = "|".join(
            f"{e.source}:{e.section_or_clause}"
            for e in evidence_pack.evidences
        )

        raw = (
            clause_text.strip()
            + intent
            + obligation_type
            + evidence_fingerprint
        )

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        """
        Compute the file path for a cache key.
        """
        return self.cache_dir / f"{key}.json"
