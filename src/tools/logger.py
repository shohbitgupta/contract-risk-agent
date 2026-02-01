import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured console logger.

    Uses stderr so MCP stdio JSON-RPC output remains clean.

    Example:
        >>> logger = setup_logger("contract-risk")
        >>> logger.info("ready")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Use stderr so MCP stdio JSON-RPC stays clean on stdout.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
