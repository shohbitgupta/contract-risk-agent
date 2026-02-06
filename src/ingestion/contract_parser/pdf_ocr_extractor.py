"""
OCR-based text extraction for scanned or image-based PDFs.

Requires system dependencies:
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Poppler (for pdf2image): e.g. `brew install poppler` on macOS

Example:
    >>> extractor = PDFOCRExtractor()
    >>> text = extractor.extract_from_file(Path("scanned.pdf"))
"""

import re
from pathlib import Path
from typing import Optional

from ingestion.contract_parser.pdf_text_extractor import PDFTextExtractionError

try:
    import pytesseract
    from pdf2image import convert_from_path
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class PDFOCRExtractor:
    """
    Extracts text from scanned/image PDFs using Tesseract OCR.

    Use when the primary extractor (pdfminer) returns insufficient text.
    """

    MIN_TEXT_LENGTH = 500  # Lower threshold for OCR output
    DPI = 200  # Higher DPI improves accuracy but is slower

    def __init__(self, lang: str = "eng"):
        if not _OCR_AVAILABLE:
            raise RuntimeError(
                "OCR dependencies not installed. "
                "Install: pip install pdf2image pytesseract. "
                "Also install Tesseract and Poppler on your system."
            )
        self.lang = lang

    def extract_from_file(self, pdf_path: Path) -> str:
        """
        Extract text by rendering each page to image and running OCR.

        Returns:
            Concatenated text from all pages, normalized.

        Raises:
            PDFTextExtractionError if OCR fails or yields too little text.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Input file is not a PDF")

        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=self.DPI,
                fmt="png",
                first_page=None,
                last_page=None,
            )
        except Exception as e:
            raise PDFTextExtractionError(
                f"Failed to convert PDF to images (is Poppler installed?): {e}"
            ) from e

        if not images:
            raise PDFTextExtractionError("PDF produced no images")

        page_texts = []
        for i, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang=self.lang)
                page_texts.append(text or "")
            except Exception as e:
                raise PDFTextExtractionError(
                    f"OCR failed on page {i + 1} (is Tesseract installed?): {e}"
                ) from e

        raw_text = "\n\n".join(page_texts)

        if not raw_text or len(raw_text.strip()) < self.MIN_TEXT_LENGTH:
            raise PDFTextExtractionError(
                "OCR produced insufficient text. The PDF may be empty or unreadable."
            )

        return self._normalize(raw_text)

    def _normalize(self, text: str) -> str:
        """Normalize OCR output while preserving structure."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(
            r"\n\s*Page\s+\d+\s+(of|/)\s+\d+\s*\n",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()


def is_ocr_available() -> bool:
    """Return True if OCR dependencies are installed and usable."""
    if not _OCR_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
