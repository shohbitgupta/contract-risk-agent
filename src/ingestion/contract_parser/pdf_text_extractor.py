
class PDFTextExtractionError(Exception):
    pass


import re
import tempfile
import requests
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


class UserContractPDFExtractor:
    """
    Extracts clean, legally safe text from Builder Buyer Agreement PDFs.
    Supports both local files and remote PDF URLs.
    """

    MIN_TEXT_LENGTH = 1000  # heuristic threshold

    # =========================================================
    # Public API
    # =========================================================

    def extract_from_url(self, pdf_url: str) -> str:
        """
        Downloads PDF from URL and extracts normalized text.
        """
        pdf_path = self._download_pdf(pdf_url)
        return self.extract_from_file(pdf_path)

    def extract_from_file(self, pdf_path: Path) -> str:
        """
        Extract text from a local PDF file.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Input file is not a PDF")

        try:
            raw_text = extract_text(str(pdf_path))
        except PDFSyntaxError as e:
            raise PDFTextExtractionError(
                f"Invalid or corrupted PDF structure: {e}"
            )

        if not raw_text or len(raw_text.strip()) < self.MIN_TEXT_LENGTH:
            raise PDFTextExtractionError(
                "PDF appears to be scanned, image-based, or contains "
                "insufficient extractable text. OCR required."
            )

        return self._normalize(raw_text)

    # =========================================================
    # Internal helpers
    # =========================================================

    def _download_pdf(self, pdf_url: str) -> Path:
        """
        Downloads PDF from URL into a temporary file.
        """
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise PDFTextExtractionError(
                f"Failed to download PDF from URL: {e}"
            )

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            raise PDFTextExtractionError(
                "URL did not return a valid PDF document"
            )

        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )
        tmp_file.write(response.content)
        tmp_file.close()

        return Path(tmp_file.name)

    def _normalize(self, text: str) -> str:
        """
        Normalize text while preserving legal structure.
        """

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (keep paragraph structure)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # Remove page number footers safely
        text = re.sub(
            r"\n\s*Page\s+\d+\s+(of|/)\s+\d+\s*\n",
            "\n",
            text,
            flags=re.IGNORECASE
        )

        # Trim trailing spaces per line
        text = "\n".join(line.rstrip() for line in text.splitlines())

        return text.strip()

