import hashlib
from pathlib import Path


def calculate_checksum(content: bytes) -> str:
    """
    Compute a SHA-256 checksum for raw bytes.

    Example:
        >>> calculate_checksum(b"hello")
        '...'
    """
    return hashlib.sha256(content).hexdigest()


def checksum_file_path(file_path: Path) -> Path:
    """
    Return the sidecar checksum path for a given file.
    """
    return file_path.with_suffix(file_path.suffix + ".checksum")


def read_existing_checksum(file_path: Path) -> str | None:
    """
    Read the checksum sidecar file if it exists.
    """
    chk_path = checksum_file_path(file_path)
    if chk_path.exists():
        return chk_path.read_text().strip()
    return None


def write_checksum(file_path: Path, checksum: str):
    """
    Write a checksum sidecar file.
    """
    chk_path = checksum_file_path(file_path)
    chk_path.write_text(checksum)
