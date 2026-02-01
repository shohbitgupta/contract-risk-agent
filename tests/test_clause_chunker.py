import textwrap

from RAG.user_contract_chunker import UserContractChunker, ChunkType


def test_clause_chunker_detects_major_sections():
    text = textwrap.dedent("""
        1. DEFINITIONS
        (a) "Act" means the Real Estate (Regulation and Development) Act, 2016.
        (b) "Allottee" means the person to whom the apartment is allotted.

        2. DELAY IN POSSESSION
        The Promoter shall hand over possession by the stated date.

        SCHEDULE-A
        1. Carpet Area: 1200 sq. ft.
    """).strip()

    chunker = UserContractChunker()
    chunks = chunker.chunk(text)

    ids = {c.chunk_id for c in chunks}
    assert "1." in ids
    assert "2." in ids
    assert "Schedule A" in ids

    schedule_chunk = next(c for c in chunks if c.chunk_id == "Schedule A")
    assert schedule_chunk.chunk_type == ChunkType.SCHEDULE
