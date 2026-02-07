from pathlib import Path
from typing import List

# -----------------------------
# Domain models
# -----------------------------
from RAG.contract_analysis import ClauseAnalysisResult
from RAG.user_contract_chunker import ContractChunk

# -----------------------------
# Chunker
# -----------------------------
from ingestion.contract_parser.contract_ingestion import UserContractChunker

# -----------------------------
# Agents
# -----------------------------
from agents.clause_understanding_agent import ClauseUnderstandingAgent
from agents.legal_explanation_agent import LegalExplanationAgent
from ingestion.contract_parser.pdf_text_extractor import UserContractPDFExtractor
from agents.contract_aggregation_agent import ContractAggregationAgent

# -----------------------------
# Retrieval
# -----------------------------
from retrieval.retrieval_orchestrator import RetrievalOrchestrator

# -----------------------------
# Vector index
# -----------------------------
from vector_index.index_registry import IndexRegistry

# -----------------------------
# Logger
# -----------------------------
from tools.logger import setup_logger

# -----------------------------
# Utils
# -----------------------------
from utils.chunk_filter import is_semantic_chunk


logger = setup_logger("contract-risk-system")

from dotenv import load_dotenv
load_dotenv()

# =========================================================
# System Orchestrator
# =========================================================

class ContractRiskAnalysisSystem:
    """
    End-to-end orchestrator for real estate contract risk analysis.
    """

    def __init__(
        self,
        index_registry: IndexRegistry,
        intent_rules_path: Path
    ):
        self.chunker = UserContractChunker()

        self.clause_agent = ClauseUnderstandingAgent(
            rules_path=intent_rules_path
        )

        self.retrieval_orchestrator = RetrievalOrchestrator(
            index_registry=index_registry
        )

        self.explanation_agent = LegalExplanationAgent()
        self.pdf_extractor = UserContractPDFExtractor()
        self.aggregation_agent = ContractAggregationAgent()

    # -----------------------------------------------------

    def analyze_contract(
        self,
        contract_text: str,
        state: str
    ) -> List[ClauseAnalysisResult]:
        """
        Run the full clause → retrieval → explanation pipeline.
        """

        logger.info("Starting contract analysis")
        logger.info(f"Target state: {state}")

        # 1️⃣ Chunk contract
        chunks: List[ContractChunk] = self.chunker.chunk(contract_text)
        logger.info(f"Generated {len(chunks)} contract chunks")

        results: List[ClauseAnalysisResult] = []

        # 2️⃣ Process each chunk independently
        for chunk in chunks:
            logger.info(f"Processing clause: {chunk.chunk_id}")
            if not is_semantic_chunk(chunk):
                logger.warning(f"Skipping non-semantic chunk: {chunk.chunk_id}")
                continue
            

            clause_result = self.clause_agent.analyze(
                clause=chunk,
                state=state
            )

            evidence_pack = self.retrieval_orchestrator.retrieve(
                clause_result=clause_result,
                state=state
            )

            clause_analysis = self.explanation_agent.explain(
                clause=chunk,
                clause_result=clause_result,
                evidence_pack=evidence_pack
            )

            results.append(clause_analysis)

        logger.info("Contract analysis completed")
        return results


# =========================================================
# CLI / Execution Entry
# =========================================================

def main(pdf_url_or_path: str) -> dict:
    """
    CLI entry to run analysis from a PDF URL or local file path.
    """

    logger.info(f"Received contract PDF: {pdf_url_or_path}")

    # -------------------------------------------------
    # 1️⃣ Extract contract text
    # -------------------------------------------------
    pdf_extractor = UserContractPDFExtractor()

    if pdf_url_or_path.startswith(("http://", "https://")):
        contract_text = pdf_extractor.extract_from_url(pdf_url_or_path)
    else:
        path = Path(pdf_url_or_path)
        if not path.is_absolute():
            base = Path(__file__).resolve().parent.parent
            path = (base / pdf_url_or_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        contract_text = pdf_extractor.extract_from_file(path)

    if not contract_text or len(contract_text.strip()) < 500:
        raise ValueError("Extracted contract text is empty or too short")

    logger.info("PDF text extraction completed")

    # -------------------------------------------------
    # 2️⃣ Load vector indexes
    # -------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent

    index_registry = IndexRegistry(
        base_dir=BASE_DIR / "src" / "data" / "vector_indexes",
        embedding_dim=384
    )
    index_registry.validate_state("uttar_pradesh")

    # -------------------------------------------------
    # 3️⃣ Load intent rules
    # -------------------------------------------------
    intent_rules_path = BASE_DIR / "src" / "configs" / "real_state_intent_rules.yaml"

    # -------------------------------------------------
    # 4️⃣ Initialize system
    # -------------------------------------------------
    system = ContractRiskAnalysisSystem(
        index_registry=index_registry,
        intent_rules_path=intent_rules_path
    )
    aggregation_agent = ContractAggregationAgent()


    # -------------------------------------------------
    # 5️⃣ Run analysis
    # -------------------------------------------------
    clause_results = system.analyze_contract(
        contract_text=contract_text,
        state="uttar_pradesh"
    )

    if not clause_results:
        raise ValueError("No clauses produced by analysis pipeline")

    analysis_details = aggregation_agent.aggregate(clause_results)

    json_dump = analysis_details.model_dump(
        mode="json",
        by_alias=True
    )

    logger.info(
        "Contract Analysis Completed | Score=%s | Grade=%s | Clauses=%d",
        analysis_details.contract_summary.overall_score,
        analysis_details.contract_summary.risk_level,
        len(analysis_details.clauses)
    )



    # -------------------------------------------------
    # 6️⃣ Console output (human readable)
    # -------------------------------------------------
    # for res in clause_results:
    #     logger.info("===================================")
    #     logger.info(f"Clause ID      : {res.clause_id}")
    #     logger.info(f"Risk Level     : {res.risk_level}")
    #     logger.info(f"Alignment      : {res.alignment}")
    #     logger.info(f"Summary        : {res.plain_summary}")
    #     logger.info(f"Action         : {res.recommended_action}")
    #     logger.info(f"Score          : {res.quality_score}")
    #     logger.info("Citations:")
    #     for c in res.citations:
    #         logger.info(f"- {c['source']} ({c['ref']})")
    #     logger.info("===================================\n")

    return json_dump


# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    import sys

    pdf_input = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "src/local_sources/F404_BBA_Shobhit Gupta.pdf"
    )

    main(pdf_input)
