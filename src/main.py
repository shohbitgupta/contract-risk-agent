from pathlib import Path
from typing import List

# -----------------------------
# Domain models
# -----------------------------
from RAG.contract_analysis import ClauseAnalysisResult
from RAG.user_contract_chunker import ContractChunk
from RAG.presentation.lawyer_summary_builder import build_lawyer_friendly_summary

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
from utils.semantic_index_evaluator import SemanticIndexEvaluator

# -----------------------------
# Configs
# -----------------------------
from configs.callibration.callibration_config_loader import CalibrationConfig

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
        intent_rules_path: Path,
        calibration_path: Path,
        state: str,
    ):
        self.chunker = UserContractChunker()

        self.clause_agent = ClauseUnderstandingAgent(
            rules_path=intent_rules_path
        )

        self.retrieval_orchestrator = RetrievalOrchestrator(
            index_registry=index_registry
        )
        self.semantic_index_evaluator = SemanticIndexEvaluator()

        self.explanation_agent = LegalExplanationAgent()
        self.pdf_extractor = UserContractPDFExtractor()
        override_path = calibration_path / "state_overrides" / f"{state.lower()}_config.yaml"
        state_override_path = override_path if override_path.exists() else None

        self.calibration_config = CalibrationConfig(
            central_path=calibration_path / "central_config.yaml",
            state_override_path=state_override_path,
        )
        self.aggregation_agent = ContractAggregationAgent(calibration=self.calibration_config)

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
            retrieval_quality = self.semantic_index_evaluator.evaluate(
                clause_result=clause_result,
                evidence_pack=evidence_pack,
                chunk=chunk,
            )

            clause_analysis = self.explanation_agent.explain(
                clause=chunk,
                clause_result=clause_result,
                evidence_pack=evidence_pack,
                retrieval_quality=retrieval_quality,
            )

            results.append(clause_analysis)

        logger.info("Contract analysis completed")
        return results


# =========================================================
# CLI / Execution Entry
# =========================================================

def main(pdf_url_or_path: str, state: str = "uttar_pradesh") -> dict:
    """
    CLI entry to run analysis from a PDF URL or local file path.
    """

    logger.info(f"Received contract PDF: {pdf_url_or_path}")
    logger.info(f"Target state: {state}")

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
    index_registry.validate_state(state)

    # -------------------------------------------------
    # 3️⃣ Load intent rules and calibration
    # -------------------------------------------------
    intent_rules_path = BASE_DIR / "src" / "configs" / "real_state_intent_rules.yaml"
    calibration_path = BASE_DIR / "src" / "configs" / "callibration"

    # -------------------------------------------------
    # 4️⃣ Initialize system
    # -------------------------------------------------
    system = ContractRiskAnalysisSystem(
        index_registry=index_registry,
        intent_rules_path=intent_rules_path,
        calibration_path=calibration_path,
        state=state,
    )


    # -------------------------------------------------
    # 5️⃣ Run analysis
    # -------------------------------------------------
    clause_results = system.analyze_contract(
        contract_text=contract_text,
        state=state,
    )

    if not clause_results:
        raise ValueError("No clauses produced by analysis pipeline")

    analysis_details = system.aggregation_agent.aggregate(clause_results)

    json_dump = analysis_details.model_dump(
        mode="json",
        by_alias=True
    )

    lawyer_summary = build_lawyer_friendly_summary(analysis_details, system.calibration_config)
    json_dump["lawyer_summary"] = lawyer_summary.model_dump(mode="json")

    logger.info(
        "Contract Analysis Completed | Score=%s | Grade=%s | Clauses=%d",
        analysis_details.contract_summary.overall_score,
        analysis_details.contract_summary.risk_level,
        len(analysis_details.clauses)
    )

    logger.info("===================================")
    logger.info("*********LAWYER SUMMARY:*********")
    logger.info(lawyer_summary.model_dump(mode="json"))
    logger.info("===================================")
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
    target_state = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "uttar_pradesh"
    )

    main(pdf_input, state=target_state)
