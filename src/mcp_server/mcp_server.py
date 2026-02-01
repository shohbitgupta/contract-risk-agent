from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agents.legal_explanation_agent import LegalExplanationAgent
from ingestion.contract_parser.pdf_text_extractor import UserContractPDFExtractor
from ingestion.contract_parser.contract_ingestion import UserContractIngestionPipeline
from retrieval.retrieval_orchestrator import RetrievalOrchestrator
from RAG.models import ExplanationResult
from tools.logger import setup_logger
from vector_index.index_registry import IndexRegistry
from agents.clause_understanding_agent import ClauseUnderstandingAgent

logger = setup_logger("mcp-server")
mcp = FastMCP("contract-risk-agent")

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data" / "vector_indexes"
CONFIG_DIR = PROJECT_ROOT / "src" / "configs"


@lru_cache(maxsize=4)
def _build_system(state: str):
    """
    Build and cache pipeline dependencies for a given state.
    """
    index_registry = IndexRegistry(
        base_dir=DATA_DIR,
        embedding_dim=384
    )
    index_registry.validate_state(state)

    intent_rules_path = CONFIG_DIR / "real_state_intent_rules.yaml"

    return {
        "clause_agent": ClauseUnderstandingAgent(rules_path=intent_rules_path),
        "retrieval": RetrievalOrchestrator(index_registry=index_registry),
        "explainer": LegalExplanationAgent(),
        "chunker": UserContractIngestionPipeline()
    }


def _to_payload(results: List[ExplanationResult]) -> List[Dict]:
    """
    Convert ExplanationResult objects into JSON-serializable dicts.
    """
    return [asdict(r) for r in results]


@mcp.tool()
def analyze_contract_pdf(pdf_url: str, state: str = "uttar_pradesh") -> Dict:
    """
    Analyze a contract PDF by URL and return explanation results.

    Example:
        >>> analyze_contract_pdf("https://example.com/contract.pdf", "uttar_pradesh")
    """
    logger.info(f"Analyzing PDF URL: {pdf_url}")
    system = _build_system(state)

    extractor = UserContractPDFExtractor()
    contract_text = extractor.extract_from_url(pdf_url)
    if not contract_text or len(contract_text.strip()) < 500:
        raise ValueError("Extracted contract text is empty or too short")

    chunks = system["chunker"].chunker.chunk(contract_text)
    results: List[ExplanationResult] = []

    for chunk in chunks:
        clause_result = system["clause_agent"].analyze(
            clause=chunk,
            state=state
        )

        evidence_pack = system["retrieval"].retrieve(
            clause_result=clause_result,
            state=state
        )

        explanation = system["explainer"].explain(
            clause=chunk,
            clause_result=clause_result,
            evidence_pack=evidence_pack
        )

        results.append(explanation)

    return {
        "state": state,
        "count": len(results),
        "results": _to_payload(results)
    }


@mcp.tool()
def analyze_contract_text(contract_text: str, state: str = "uttar_pradesh") -> Dict:
    """
    Analyze raw contract text and return explanation results.

    Example:
        >>> analyze_contract_text("Clause text...", "uttar_pradesh")
    """
    logger.info("Analyzing provided contract text")
    system = _build_system(state)
    chunks = system["chunker"].chunker.chunk(contract_text)

    results: List[ExplanationResult] = []
    for chunk in chunks:
        clause_result = system["clause_agent"].analyze(
            clause=chunk,
            state=state
        )

        logger.info(f"Clause agent result : {clause_result}")

        evidence_pack = system["retrieval"].retrieve(
            clause_result=clause_result,
            state=state
        )

        logger.info(f"retrieval agent result : {evidence_pack}")

        explanation = system["explainer"].explain(
            clause=chunk,
            clause_result=clause_result,
            evidence_pack=evidence_pack
        )

        logger.info(f"explanation agent result : {explanation}")
        results.append(explanation)

    return {
        "state": state,
        "count": len(results),
        "results": _to_payload(results)
    }


if __name__ == "__main__":
    mcp.run()
