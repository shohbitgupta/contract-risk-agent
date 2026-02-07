import re
from RAG.user_contract_chunker import ContractChunk

LEGAL_KEYWORDS = {
    "shall", "must", "may", "agree", "entitled", "liable", "obligated",
    "subject to", "provided that", "in accordance with",
    "refund", "interest", "possession", "delay",
    "termination", "cancellation", "jurisdiction",
    "penalty", "compensation", "defect", "warranty"
}

NUMBER_ONLY_PATTERN = re.compile(r"^[\d\W]+$")


def is_semantic_chunk(chunk: ContractChunk) -> bool:
    """
    Conservative legal-semantic filter.
    Returns False ONLY when we are very confident the chunk
    is not legally meaningful.
    """

    text = chunk.text.strip()
    lower = text.lower()

    # 1️⃣ Extremely short chunks are never semantic
    if len(text) < 25:
        return False

    # 2️⃣ Pure numbering / punctuation
    if NUMBER_ONLY_PATTERN.match(text):
        return False

    # 3️⃣ Headings only (no verbs)
    if (
        text.isupper()
        and len(text.split()) <= 4
        and not any(k in lower for k in LEGAL_KEYWORDS)
    ):
        return False

    # 4️⃣ Single token bullets like "(a)", "(i)"
    if re.fullmatch(r"\([a-zivx]+\)", lower):
        return False

    # 5️⃣ Keep if any legal marker is present
    if any(k in lower for k in LEGAL_KEYWORDS):
        return True

    # 6️⃣ Fallback: keep if sentence-like
    has_verb = any(v in lower for v in [" shall ", " must ", " may ", " agrees "])
    return has_verb
