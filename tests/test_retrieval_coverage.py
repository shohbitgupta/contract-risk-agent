from pathlib import Path

import pytest


def _has_vector_indexes(base_dir: Path) -> bool:
    return (base_dir / "uttar_pradesh").exists()


def test_retrieval_returns_evidence_for_possession_delay():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")

    from agents.clause_understanding_agent import ClauseUnderstandingAgent
    from retrieval.retrieval_orchestrator import RetrievalOrchestrator
    from RAG.user_contract_chunker import ContractChunk, ChunkType
    from vector_index.index_registry import IndexRegistry

    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "src" / "data" / "vector_indexes"
    if not _has_vector_indexes(base_dir):
        pytest.skip("Vector indexes not present for test run.")

    intent_rules_path = project_root / "src" / "configs" / "real_state_intent_rules.yaml"
    clause_text = (
        "The Promoter shall hand over possession by the agreed date. "
        "In case of delay in possession, the allottee is entitled to remedies."
    )

    clause = ContractChunk(
        chunk_id="2.1",
        text=clause_text,
        chunk_type=ChunkType.CLAUSE
    )

    clause_agent = ClauseUnderstandingAgent(rules_path=intent_rules_path)
    clause_result = clause_agent.analyze(clause=clause, state="uttar_pradesh")

    registry = IndexRegistry(base_dir=base_dir, embedding_dim=384)
    orchestrator = RetrievalOrchestrator(index_registry=registry)
    evidence_pack = orchestrator.retrieve(clause_result=clause_result, state="uttar_pradesh")

    assert len(evidence_pack.evidences) > 0
    assert all(ev.text for ev in evidence_pack.evidences)
