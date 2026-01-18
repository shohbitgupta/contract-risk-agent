from pathlib import Path
import re
import time
import requests

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

from pdf2image import convert_from_path
import pytesseract

from tools.logger import setup_logger

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path("data/sources/uttar_pradesh")
TMP_DIR = BASE_DIR / "_tmp_pdfs"

logger = setup_logger("contract-risk-system")

# Ordered list of mirrors per document
PDF_SOURCES = {
    "rera_act_2016.txt": [
        "https://up-rera.in/pdf/reraact.pdf",
        "https://www.icsi.edu/media/portals/86/bare%20acts/THE%20REAL%20ESTATE%20%28REGULATION%20AND%20DEVELOPMENT%29%20ACT%2C%202016.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2158/1/a2016-16.pdf",
    ],
    "up_rera_rules_2016.txt": [
        "https://www.awasbandhu.in/wp-content/uploads/2018/06/UP-RERA-Rules-2016.pdf",
    ],
    "model_bba_form_l.txt": [
        "https://up-rera.in/ViewDocument?Param=PRJ33967944BBA.pdf",
    ],
}

MIN_TEXT_LENGTH = 1000  # hard safety threshold


# =========================================================
# DOWNLOAD (ROBUST)
# =========================================================

def download_pdf(url: str, target: Path, retries: int = 3):
    if target.exists() and target.stat().st_size > 1_000_000:
        logger.info(f"Using cached PDF: {target.name}")
        return

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading (attempt {attempt}): {url}")
            with requests.get(url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except requests.exceptions.RequestException as e:
            logger.info(f"Download failed: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Safe normalization for legal documents.
    Preserves section / rule structure.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page numbers
    text = re.sub(
        r"\n\s*Page\s+\d+\s*(of\s+\d+)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix broken lines inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()


# =========================================================
# TEXT EXTRACTION (PDFMINER)
# =========================================================

def extract_pdf_text(pdf_path: Path) -> str:
    try:
        text = extract_text(str(pdf_path))
    except PDFSyntaxError:
        text = ""

    if text and len(text.strip()) >= MIN_TEXT_LENGTH:
        return normalize_text(text)

    raise RuntimeError("Text extraction insufficient, OCR required")


# =========================================================
# OCR EXTRACTION (RULES-SAFE)
# =========================================================

def extract_pdf_text_with_ocr(pdf_path: Path) -> str:
    logger.info(f"OCR extraction started for {pdf_path.name}")

    images = convert_from_path(
        pdf_path,
        dpi=300,
        fmt="png"
    )

    if not images:
        raise RuntimeError("PDF → image conversion failed")

    text_parts = []

    for img in images:
        page_text = pytesseract.image_to_string(
            img,
            lang="eng",
            config="--psm 6"
        )
        if page_text.strip():
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    if len(full_text.strip()) < MIN_TEXT_LENGTH:
        raise RuntimeError("OCR text too short — extraction unsafe")

    return normalize_text(full_text)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    logger.info("Preparing UP-RERA legal source files")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    for output_name, urls in PDF_SOURCES.items():
        pdf_path = TMP_DIR / f"{output_name}.pdf"
        txt_path = BASE_DIR / output_name

        last_error = None

        for url in urls:
            try:
                download_pdf(url, pdf_path)
                break
            except Exception as e:
                last_error = e
                logger.info("Source failed, trying next mirror…")

        if not pdf_path.exists():
            raise RuntimeError(
                f"All download sources failed for {output_name}"
            ) from last_error

        logger.info(f"Extracting text → {output_name}")

        if output_name == "up_rera_rules_2016.txt":
            text = extract_pdf_text_with_ocr(pdf_path)
        else:
            try:
                text = extract_pdf_text(pdf_path)
            except RuntimeError:
                text = extract_pdf_text_with_ocr(pdf_path)

        txt_path.write_text(text, encoding="utf-8")
        logger.info(f"Saved: {txt_path}")

    logger.info("\nAll UP-RERA source files prepared successfully.")
    logger.info("You may now run the ingestion pipeline.")


if __name__ == "__main__":
    main()
