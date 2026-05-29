import logging
import sys
from pathlib import Path

_configured = False


def setup_logging(log_level: str = "INFO", log_file: str = "logs/chat.log") -> None:
    global _configured
    if _configured:
        return

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — everything DEBUG and above
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — WARNING and above only (so it doesn't pollute chat output)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)

    root = logging.getLogger("llm")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.addHandler(fh)
    root.addHandler(sh)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"llm.{name}")