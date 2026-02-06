
class PDFTextExtractionError(Exception):
    """
    Raised when PDF extraction fails or yields unusable content.

    Example:
        >>> raise PDFTextExtractionError("Scanned PDF requires OCR")
    """


import logging
import re
import tempfile
import requests
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

logger = logging.getLogger("pdf-extractor")


class UserContractPDFExtractor:
    """
    Extracts clean, legally safe text from Builder Buyer Agreement PDFs.
    Supports both local files and remote PDF URLs.

    Example:
        >>> extractor = UserContractPDFExtractor()
        >>> text = extractor.extract_from_url("https://example.com/contract.pdf")
    """

    MIN_TEXT_LENGTH = 1000  # heuristic threshold

    # =========================================================
    # Public API
    # =========================================================

    def extract_from_url(self, pdf_url: str) -> str:
        """
        Downloads PDF from URL and extracts normalized text.

        Returns:
            Extracted text as a single string.
        """
        pdf_path = self._download_pdf(pdf_url)
        return self.extract_from_file(pdf_path)

    def extract_from_file(
        self,
        pdf_path: Path,
        use_ocr_if_scanned: bool = True,
    ) -> str:
        """
        Extract text from a local PDF file.

        Tries text extraction first; if the PDF appears scanned (insufficient
        text), falls back to OCR when use_ocr_if_scanned is True.

        Raises:
            PDFTextExtractionError if extraction fails or text is too short.
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

        if raw_text and len(raw_text.strip()) >= self.MIN_TEXT_LENGTH:
            return self._normalize(raw_text)

        if use_ocr_if_scanned:
            logger.info(
                "Text layer insufficient; attempting OCR (scanned/image PDF)."
            )
            raw_text = self._extract_via_ocr(pdf_path)
            if raw_text:
                logger.info("OCR extraction succeeded.")
                return self._normalize(raw_text)
            logger.warning("OCR unavailable or failed; install pdf2image, pytesseract, Tesseract, and Poppler to enable.")

        raise PDFTextExtractionError(
            "PDF appears to be scanned, image-based, or contains "
            "insufficient extractable text. OCR required (install pdf2image, "
            "pytesseract, Tesseract, and Poppler to enable automatic OCR)."
        )

    def _extract_via_ocr(self, pdf_path: Path) -> str:
        """
        Extract text using OCR when the PDF has no usable text layer.

        Returns:
            Extracted text, or empty string if OCR is unavailable or fails.
        """
        try:
            from ingestion.contract_parser.pdf_ocr_extractor import (
                PDFOCRExtractor,
                is_ocr_available,
            )
        except ImportError:
            return ""

        if not is_ocr_available():
            return ""

        try:
            ocr = PDFOCRExtractor()
            return ocr.extract_from_file(pdf_path)
        except Exception:
            return ""

    # =========================================================
    # Internal helpers
    # =========================================================

    def _download_pdf(self, pdf_url: str) -> Path:
        """
        Downloads PDF from URL into a temporary file.

        Returns:
            Path to the downloaded temp PDF.
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

        This trims extra whitespace and removes page footers.
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

