from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from tools.logger import setup_logger
from tools.checksum import (
    calculate_checksum,
    read_existing_checksum,
    write_checksum
)

logger = setup_logger("pdf_crawler")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "contract-risk-agent/1.0"
})


def crawl_and_download(base_url: str, save_dir: Path):
    """
    Crawl a web page and download all linked PDFs.

    Returns:
        Dict with downloaded/skipped/failed lists.

    Example:
        >>> crawl_and_download("https://example.com/docs", Path("data/pdfs"))
    """
    logger.info(f"Starting crawl: {base_url}")

    results = {
        "base_url": base_url,
        "downloaded": [],
        "skipped": [],
        "failed": []
    }

    # Fetch HTML page
    try:
        response = SESSION.get(base_url, timeout=30)
        response.raise_for_status()
    except RequestException as e:
        logger.error(f"Failed to fetch page: {base_url} | {e}")
        results["failed"].append({"error": str(e)})
        return results

    soup = BeautifulSoup(response.text, "html.parser")

    pdf_urls = [
        urljoin(base_url, link["href"])
        for link in soup.find_all("a", href=True)
        if link["href"].lower().endswith(".pdf")
    ]

    if not pdf_urls:
        logger.warning(f"No PDF links found on page: {base_url}")
        results["failed"].append({"error": "No PDF links found"})
        return results

    # Ensure directory exists
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.critical(f"Cannot create directory {save_dir}: {e}")
        results["failed"].append({"error": str(e)})
        return results

    # Download PDFs
    for pdf_url in pdf_urls:
        file_name = pdf_url.split("/")[-1]
        file_path = save_dir / file_name

        try:
            resp = SESSION.get(pdf_url, timeout=30)
            resp.raise_for_status()

            if "application/pdf" not in resp.headers.get("Content-Type", ""):
                raise ValueError("URL did not return a PDF")

            new_checksum = calculate_checksum(resp.content)
            old_checksum = read_existing_checksum(file_path)

            if file_path.exists() and old_checksum == new_checksum:
                logger.info(f"Skipped (unchanged): {file_name}")
                results["skipped"].append(str(file_path))
                continue

            file_path.write_bytes(resp.content)
            write_checksum(file_path, new_checksum)

            logger.info(f"Downloaded: {file_name}")
            results["downloaded"].append(str(file_path))

        except RequestException as e:
            logger.error(f"Network error while downloading {pdf_url}: {e}")
            results["failed"].append({
                "url": pdf_url,
                "error": str(e)
            })

        except Exception as e:
            logger.error(f"Error processing {pdf_url}: {e}")
            results["failed"].append({
                "url": pdf_url,
                "error": str(e)
            })

    logger.info(
        f"Crawl finished | downloaded={len(results['downloaded'])}, "
        f"skipped={len(results['skipped'])}, failed={len(results['failed'])}"
    )

    return results


def download_single_pdf(pdf_url: str, save_dir: Path):
    """
    Download a single PDF and write it to disk with checksum.

    Returns:
        Dict with status and file path.
    """
    # Registry location
    file_name = pdf_url.split("=")[-1]
    file_path = save_dir / file_name

    logger.info(f"Downloading single PDF: {pdf_url} at: file_path {file_path}")

    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        response = SESSION.get(pdf_url, timeout=30)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            raise ValueError("URL did not return a PDF")

        new_checksum = calculate_checksum(response.content)
        old_checksum = read_existing_checksum(file_path)

        if file_path.exists() and old_checksum == new_checksum:
            logger.info(f"Skipped (unchanged): {file_name}")
            return {
                "status": "skipped",
                "file": str(file_path)
            }

        file_path.write_bytes(response.content)
        write_checksum(file_path, new_checksum)

        logger.info(f"Downloaded: {file_name}")
        return {
            "status": "downloaded",
            "file": str(file_path),
            "checksum": new_checksum
        }

    except (PermissionError, OSError) as e:
        logger.critical(f"Filesystem error for {file_name}: {e}")
        return {
            "status": "error",
            "url": pdf_url,
            "error": str(e)
        }

    except RequestException as e:
        logger.error(f"Network error for {pdf_url}: {e}")
        return {
            "status": "error",
            "url": pdf_url,
            "error": str(e)
        }

    except Exception as e:
        logger.error(f"Unexpected error for {pdf_url}: {e}")
        return {
            "status": "error",
            "url": pdf_url,
            "error": str(e)
        }
