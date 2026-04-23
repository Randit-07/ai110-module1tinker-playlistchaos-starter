import logging
import logging.handlers
import os
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger once. Subsequent calls are no-ops.

    INFO+ goes to logs/agent.log (rotating, 1MB x 3).
    WARNING+ also goes to stderr so problems surface during interactive runs.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("playlistchaos")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "agent.log")

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    _CONFIGURED = True
    logger = logging.getLogger("playlistchaos")
    logger.info("logging initialized: file=%s level=%s", log_path, logging.getLevelName(level))
    return logger
