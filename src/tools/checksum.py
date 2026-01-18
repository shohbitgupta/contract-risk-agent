import hashlib
from pathlib import Path


def calculate_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def checksum_file_path(file_path: Path) -> Path:
    return file_path.with_suffix(file_path.suffix + ".checksum")


def read_existing_checksum(file_path: Path) -> str | None:
    chk_path = checksum_file_path(file_path)
    if chk_path.exists():
        return chk_path.read_text().strip()
    return None


def write_checksum(file_path: Path, checksum: str):
    chk_path = checksum_file_path(file_path)
    chk_path.write_text(checksum)
