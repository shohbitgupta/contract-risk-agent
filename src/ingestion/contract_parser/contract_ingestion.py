from pathlib import Path

from ingestion.contract_parser.pdf_text_extractor import (
    UserContractPDFExtractor,
    PDFTextExtractionError
)
from RAG.user_contract_chunker import UserContractChunker, ContractChunk


class UserContractIngestionError(Exception):
    """
    Raised when contract ingestion fails for any reason.

    Example:
        >>> raise UserContractIngestionError("No clauses extracted")
    """


class UserContractIngestionPipeline:
    """
    Wires:
    PDF → clean text → clause-level chunks

    Example:
        >>> pipeline = UserContractIngestionPipeline()
        >>> chunks = pipeline.ingest(Path("contract.pdf"))
    """

    def __init__(self):
        self.extractor = UserContractPDFExtractor()
        self.chunker = UserContractChunker()

    def ingest(self, pdf_path: Path) -> list[ContractChunk]:
        """
        Main entry point for user contract ingestion.

        Returns:
            List of ContractChunk objects.
        """

        try:
            # 1. Extract clean text from PDF
            text = self.extractor.extract_from_file(pdf_path)

            # 2. Chunk into clause-level objects
            chunks = self.chunker.chunk(text)

            if not chunks:
                raise UserContractIngestionError(
                    "No clauses could be extracted from contract."
                )

            return chunks

        except PDFTextExtractionError as e:
            raise UserContractIngestionError(
                f"PDF extraction failed: {e}"
            )

        except Exception as e:
            raise UserContractIngestionError(
                f"Contract ingestion failed: {e}"
            )
