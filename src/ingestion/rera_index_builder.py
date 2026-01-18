# build_up_rera_indexes.py

from pathlib import Path
from typing import List
import re
import numpy as np

from vector_index.embedding import EmbeddingGenerator
from vector_index.faiss_index import FAISSVectorIndex
from vector_index.index_base import IndexDocument

from tools.logger import setup_logger

logger = setup_logger("up-rera-index-builder")

# =========================================================
# CONFIG
# =========================================================

STATE = "uttar_pradesh"
EMBEDDING_DIM = 384

SOURCE_BASE = Path("data/sources") / STATE
INDEX_BASE = Path("data/vector_indexes") / STATE

LEGAL_SOURCES = {
    "rera_act": SOURCE_BASE / "rera_act_2016.txt",
    "rera_rules": SOURCE_BASE / "up_rera_rules_2016.txt",
    "model_bba": SOURCE_BASE / "model_bba_form_l.txt",
    "circulars": SOURCE_BASE / "circulars",
    "case_law": SOURCE_BASE / "case_law",
}

# =========================================================
# CHUNKING (LEGAL SAFE)
# =========================================================

def chunk_legal_text(text: str, source: str) -> List[IndexDocument]:
    chunks: List[IndexDocument] = []

    splits = re.split(
        r"\n(?=(Section\s+\d+|Rule\s+\d+|Clause\s+\d+|\d+\.\s))",
        text
    )

    for i, part in enumerate(splits):
        content = part.strip()
        if len(content) < 200:
            continue

        chunks.append(
            IndexDocument(
                content=content,
                metadata={
                    "source": source,
                    "chunk_id": f"{source}_{i}"
                }
            )
        )

    return chunks


# =========================================================
# LOADERS
# =========================================================

def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path.read_text(encoding="utf-8")


def load_directory(dir_path: Path) -> List[IndexDocument]:
    docs: List[IndexDocument] = []

    for file in sorted(dir_path.glob("*.txt")):
        logger.info(f"Loading {file.name}")
        text = load_text_file(file)
        docs.extend(chunk_legal_text(text, file.stem))

    return docs


# =========================================================
# BUILD INDEX
# =========================================================

def build_index(
    index_name: str,
    documents: List[IndexDocument],
    embedding_model: EmbeddingGenerator
):
    if not documents:
        logger.warning(f"No documents found for {index_name}, skipping")
        return

    logger.info(f"Building index: {index_name}")
    logger.info(f"Chunks: {len(documents)}")

    texts = [doc.content for doc in documents]

    embeddings = embedding_model.embed(texts)

    # Ensure numpy + float32
    embeddings = np.array(embeddings, dtype="float32")

    index_path = INDEX_BASE / f"{index_name}.faiss"
    index = FAISSVectorIndex(
        index_path=index_path,
        dim=EMBEDDING_DIM
    )

    index.add(
        embeddings=embeddings,
        documents=documents
    )

    index.persist()
    logger.info(f"Index written: {index_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    logger.info("Starting UP-RERA vector index ingestion")

    INDEX_BASE.mkdir(parents=True, exist_ok=True)

    embedding_model = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2"
    )

    # -----------------------------
    # RERA ACT
    # -----------------------------
    act_docs = chunk_legal_text(
        load_text_file(LEGAL_SOURCES["rera_act"]),
        "rera_act"
    )
    build_index("rera_act", act_docs, embedding_model)

    # -----------------------------
    # RERA RULES
    # -----------------------------
    rules_docs = chunk_legal_text(
        load_text_file(LEGAL_SOURCES["rera_rules"]),
        "rera_rules"
    )
    build_index("rera_rules", rules_docs, embedding_model)

    # -----------------------------
    # MODEL BBA
    # -----------------------------
    bba_docs = chunk_legal_text(
        load_text_file(LEGAL_SOURCES["model_bba"]),
        "model_bba"
    )
    build_index("model_bba", bba_docs, embedding_model)

    # -----------------------------
    # CIRCULARS
    # -----------------------------
    circular_docs = load_directory(LEGAL_SOURCES["circulars"])
    build_index("circulars", circular_docs, embedding_model)

    # -----------------------------
    # CASE LAW
    # -----------------------------
    case_law_docs = load_directory(LEGAL_SOURCES["case_law"])
    build_index("case_law", case_law_docs, embedding_model)

    logger.info("UP-RERA vector index ingestion completed successfully")


if __name__ == "__main__":
    main()
