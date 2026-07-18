"""Logging setup shared by all scripts."""

from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging", "get_logger"]

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FMT, datefmt="%H:%M:%S", stream=sys.stdout, force=True,
    )


def get_logger(name):
    return logging.getLogger(name)
