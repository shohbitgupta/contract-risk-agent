from pathlib import Path
import sys

# Allow running as: `python src/scripts/rebuild_statute_index.py`
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.statute_section_indexer import StatuteSectionIndexer  # noqa: E402


STATE = "uttar_pradesh"

# Prepared by: `python src/scripts/prepare_rera_sources.py`
STATUTE_FILE = BASE_DIR / "src" / "data" / "sources" / STATE / "rera_act_2016.txt"

# Runtime expects: src/data/vector_indexes/<state>/*.faiss
INDEX_OUTPUT_DIR = BASE_DIR / "src" / "data" / "vector_indexes" / STATE


def main():
    with open(STATUTE_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    indexer = StatuteSectionIndexer(
        act_name="RERA Act, 2016",
        doc_type="rera_act",
        state="central"
    )

    index_path, n = indexer.build_index(
        full_text=full_text,
        output_dir=INDEX_OUTPUT_DIR,
        index_name="rera_act",
        source="RERA Act, 2016",
        delete_existing=True,
    )
    print(f"✅ Rebuilt statute index: {index_path} ({n} sections)")


if __name__ == "__main__":
    main()