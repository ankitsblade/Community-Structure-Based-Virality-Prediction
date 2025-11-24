# microviral/logger.py

import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "microviral.log")

# Create logger
logger = logging.getLogger("microviral")
logger.setLevel(LOG_LEVEL)

# Prevent duplicate handlers if module is imported multiple times
if not logger.handlers:

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(LOG_LEVEL)

    # Formatter (timestamp + module + level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
