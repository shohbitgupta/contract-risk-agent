"""
Validate statute index quality (section-level chunking + metadata).

Usage:
  python src/scripts/validate_statute_index.py

What it checks:
  - Index file exists at src/data/vector_indexes/<state>/rera_act.faiss
  - Documents contain required metadata keys
  - Section headings look like "Section N"
  - Sample semantic queries retrieve clean sections (prints top matches)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vector_index.embedding import EmbeddingGenerator  # noqa: E402
from vector_index.faiss_index import FAISSVectorIndex  # noqa: E402
from vector_index.index_base import IndexDocument  # noqa: E402


STATE = "uttar_pradesh"
INDEX_NAME = "rera_act"
INDEX_PATH = BASE_DIR / "src" / "data" / "vector_indexes" / STATE / f"{INDEX_NAME}.faiss"


def _assert_doc_quality(doc: IndexDocument) -> None:
    # IndexDocument already enforces: source + chunk_id; content length.
    meta = doc.metadata

    for k in ("doc_type", "act", "section", "section_number", "state", "index_name"):
        if k not in meta:
            raise AssertionError(f"Missing metadata key: {k}")

    sec = str(meta.get("section") or "")
    if not sec.lower().startswith("section "):
        raise AssertionError(f"Bad section label: {sec!r}")

    # Anchor-friendly: content should visibly contain the section label early.
    content_head = doc.content.strip().lower()[:80]
    if "section" not in content_head:
        raise AssertionError("Content does not look like a section block")


def _search(index: FAISSVectorIndex, embedder: EmbeddingGenerator, query: str, top_k: int = 5) -> List[IndexDocument]:
    q = np.asarray(embedder.embed([query])[0], dtype="float32")
    return index.search(query_embedding=q, top_k=top_k)


def main() -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. "
            "Run: python src/scripts/rebuild_statute_index.py"
        )

    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    index = FAISSVectorIndex.load(index_path=INDEX_PATH, dim=384)

    docs = list(index.documents.values())
    if not docs:
        raise RuntimeError("Index is empty (no documents).")

    # 1) Metadata + section-shape checks
    bad = 0
    for d in docs[:200]:  # sample first N (fast)
        try:
            _assert_doc_quality(d)
        except AssertionError as e:
            bad += 1
            print(f"❌ Doc failed: chunk_id={d.metadata.get('chunk_id')} error={e}")

    print(f"Checked {min(len(docs), 200)} docs; failures={bad}")

    # 2) Retrieval spot checks (prints top hits)
    queries = [
        "Section 18 refund RERA",
        "Section 14 defect liability RERA",
        "Section 6 force majeure RERA",
        "return of amount and compensation promoter fails to complete",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        results = _search(index, embedder, q, top_k=5)
        for i, r in enumerate(results, start=1):
            meta = r.metadata
            print(
                f"{i}. {meta.get('section')} | title={meta.get('title')!r} | chunk_id={meta.get('chunk_id')}"
            )
            print(r.content.strip().splitlines()[0][:160])

    print("\n✅ Validation complete.")


if __name__ == "__main__":
    main()

