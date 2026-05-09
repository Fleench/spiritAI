"""Shared filesystem paths derived from the configured workspace root."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WORKSPACE_PATH = Path(os.getenv("WORKSPACE_PATH", "/workspace")).expanduser()


def workspace_path(*subpaths: str) -> Path:
    """Return a path inside the configured workspace root."""
    return WORKSPACE_PATH.joinpath(*subpaths)
