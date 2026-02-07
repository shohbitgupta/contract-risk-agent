import logging
import sys
from typing import Optional

# Use level_style for colored+emoji; fallback to levelname when color disabled
LOG_FORMAT = "%(asctime)s | %(level_style)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes (only applied when output is a TTY)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
# 256-color palette (\033[38;5;Nm)
SEA_GREEN = "\033[38;5;72m"   # medium sea green
BROWN = "\033[38;5;94m"       # dark yellow / brown
BLOOD_RED = "\033[38;5;124m"  # dark red
# Cyan shades for log messages (level name stays distinct; message in cyan)
CYAN_MSG_INFO = "\033[38;5;51m"     # bright cyan
CYAN_MSG_WARNING = "\033[38;5;37m"  # medium cyan
CYAN_MSG_ERROR = "\033[38;5;36m"    # cyan
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# Level → (level_color, emoji, message_color)
LEVEL_STYLES = {
    logging.DEBUG: (DIM + CYAN, "🔍", DIM + CYAN),
    logging.INFO: (SEA_GREEN, "ℹ️ ", CYAN_MSG_INFO),
    logging.WARNING: (BROWN, "⚠️ ", CYAN_MSG_WARNING),
    logging.ERROR: (BLOOD_RED + BOLD, "❌", CYAN_MSG_ERROR),
    logging.CRITICAL: (BLOOD_RED + BOLD, "🔥", CYAN_MSG_ERROR),
}


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors and emojis per log level.
    Disables colors when stderr is not a TTY (e.g. in CI or pipes).
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_color: Optional[bool] = None,
    ):
        super().__init__(fmt=fmt or LOG_FORMAT, datefmt=datefmt or DATE_FORMAT)
        if use_color is None:
            use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if self.use_color and record.levelno in LEVEL_STYLES:
            level_color, emoji, msg_color = LEVEL_STYLES[record.levelno]
            record.level_style = f"{emoji} {level_color}{record.levelname:<8}{RESET}"
            record.message = f"{msg_color}{msg}{RESET}"
        else:
            record.level_style = f"{record.levelname:<10}"
            record.message = msg
        return super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured console logger with colored output and emojis per level.

    - DEBUG: dim cyan, 🔍
    - INFO: sea green, ℹ️
    - WARNING: brown, ⚠️
    - ERROR: blood red (bold), ❌
    - CRITICAL: blood red (bold), 🔥

    Colors/emojis are disabled when stderr is not a TTY (e.g. CI, pipes).

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

    formatter = ColoredFormatter()

    # Use stderr so MCP stdio JSON-RPC stays clean on stdout.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
