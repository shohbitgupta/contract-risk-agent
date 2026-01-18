import json
from pathlib import Path

from tools import pdf_crawler
from tools.logger import setup_logger

logger = setup_logger("ingestion_pipeline")


def run_ingestion(state: str):
    """
        Runs legal document ingestion for a given state.
        Downloads central + state-specific RERA documents
        into project-local data directory.
        """

    # --------------------------------------------------
    # 1. Resolve paths safely
    # --------------------------------------------------

    # Project root = contract-risk-agent/
    project_root = Path(__file__).resolve().parents[2]

    logger.info(f"Ingestion started for {state} at {project_root}")

    # Registry location
    registry_path = (
            Path(__file__).resolve().parent / "config" / "registry.json"
    )

    # Base data directory (WRITE SAFE)
    base_data_dir = project_root / "data" / "rera_docs" / state

    # --------------------------------------------------
    # 2. Load registry
    # --------------------------------------------------

    with open(registry_path, "r") as json_file:
        registry = json.load(json_file)

    india_registry = registry["india"]

    # --------------------------------------------------
    # 3. Validate state
    # --------------------------------------------------

    if state not in india_registry:
        raise ValueError(
            f"State '{state}' not found in registry. "
            f"Available: {list(india_registry.keys())}"
        )

    central_docs = india_registry["central"]
    state_docs = india_registry[state]

    # --------------------------------------------------
    # 4. Ingest CENTRAL documents
    # --------------------------------------------------

    central_dir = base_data_dir / "central"

    for doc in central_docs:
        url = doc["url"]

        if url.lower().endswith(".pdf"):
            pdf_crawler.download_single_pdf(url, central_dir)
        else:
            pdf_crawler.crawl_and_download(url, central_dir)

    # --------------------------------------------------
    # 5. Ingest STATE-SPECIFIC documents
    # --------------------------------------------------

    state_dir = base_data_dir / "state"

    for doc in state_docs:
        url = doc["url"]

        if url.lower().endswith(".pdf"):
            pdf_crawler.download_single_pdf(url, state_dir)
        else:
            pdf_crawler.crawl_and_download(url, state_dir)
