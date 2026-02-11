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
# CHUNKING (LEGAL SAFE) – section labels & RERA phrasing
# =========================================================

# Act: "18. (1) If the promoter..." or "(2) The promoter..."
_ACT_MAIN = re.compile(r"^(\d+)\.\s*(\(\d+\))?\s*", re.MULTILINE)
_ACT_SUB = re.compile(r"^\s*(\(\d+\))\s+", re.MULTILINE)
# Rules: "15- The Authority..." or "Rule 16" style
_RULE_HEAD = re.compile(r"^(\d+)[-.)]\s*", re.MULTILINE)
_SECTION_WORD = re.compile(r"^(Section|Rule|Clause)\s+(\d+[A-Za-z]*(?:\(\d+\))?)", re.IGNORECASE)


def _normalize_act_section(content: str, source: str) -> tuple[str, str]:
    """Extract Section N or Section N(k) and return (section_id, content_with_prefix)."""
    content = content.strip()
    main = _ACT_MAIN.match(content)
    if main:
        num, sub = main.group(1), main.group(2)
        section_id = f"Section {num}{sub or ''}".strip()
        # Prefix so embedding sees "Section 18(1)" in text (aligns with intent rules)
        prefix = f"{section_id}. "
        if not content.startswith(prefix) and not content.startswith(section_id):
            content = prefix + content[main.end() :].strip()
        return section_id, content
    sub = _ACT_SUB.match(content)
    if sub:
        # Continuation of same section e.g. "(2) The promoter..."
        section_id = f"Section {sub.group(1)}"  # "(2)" -> Section (2); we need parent
        return section_id, content
    word = _SECTION_WORD.match(content)
    if word:
        section_id = f"{word.group(1)} {word.group(2)}"
        return section_id, content
    return "", content


def _normalize_rule_section(content: str) -> tuple[str, str]:
    """Extract Rule N from content."""
    content = content.strip()
    m = _RULE_HEAD.match(content)
    if m:
        section_id = f"Rule {m.group(1)}"
        prefix = f"{section_id}. "
        rest = content[m.end() :].strip()
        if not rest.startswith(prefix):
            content = prefix + rest
        return section_id, content
    word = _SECTION_WORD.match(content)
    if word:
        section_id = f"{word.group(1)} {word.group(2)}"
        return section_id, content
    return "", content


def chunk_legal_text(
    text: str,
    source: str,
    *,
    doc_type: str,
    state: str | None
) -> List[IndexDocument]:
    chunks: List[IndexDocument] = []
    is_act = (doc_type == "rera_act")
    is_rules = (doc_type == "state_rule")

    # Split: Act by main section number only; Rules by "N-" or "N-(k)" (rule numbers often mid-line in UP rules).
    if is_act:
        splits = re.split(r"\n(?=\d+\.\s)", text)
    elif is_rules:
        # Rule numbers appear mid-line (e.g. "Rate of interest 15- The Authority"); split on pattern, not only after newline.
        splits = re.split(
            r"(?=\d+-\s*(?:\(\d+\)|\s+[A-Z]))",
            text
        )
    else:
        splits = re.split(
            r"\n(?=(Section\s+\d+|Rule\s+\d+|Clause\s+\d+|\d+\.\s))",
            text
        )

    for i, part in enumerate(splits):
        raw = part.strip()
        if len(raw) < 100:
            continue

        section_id = ""
        content = raw
        if is_act:
            section_id, content = _normalize_act_section(raw, source)
        elif is_rules:
            section_id, content = _normalize_rule_section(raw)
        else:
            section_match = re.match(
                r"^(Section|Rule|Clause)\s+\d+[A-Za-z]*(?:\(\d+\))?",
                content,
                re.IGNORECASE
            )
            if section_match:
                section_id = section_match.group(0)

        if len(content) < 50:
            continue

        chunks.append(
            IndexDocument(
                content=content,
                metadata={
                    "source": source,
                    "chunk_id": f"{source}_{i}",
                    "doc_type": doc_type,
                    "state": state,
                    "section": section_id or None
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
        "rera_act",
        doc_type="rera_act",
        state=None
    )
    build_index("rera_act", act_docs, embedding_model)

    # -----------------------------
    # RERA RULES
    # -----------------------------
    rules_docs = chunk_legal_text(
        load_text_file(LEGAL_SOURCES["rera_rules"]),
        "rera_rules",
        doc_type="state_rule",
        state=STATE
    )
    build_index("rera_rules", rules_docs, embedding_model)

    # -----------------------------
    # MODEL BBA
    # -----------------------------
    bba_docs = chunk_legal_text(
        load_text_file(LEGAL_SOURCES["model_bba"]),
        "model_bba",
        doc_type="model_agreement",
        state=STATE
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
