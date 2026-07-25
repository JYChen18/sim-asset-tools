"""Prepare portable, simulation-ready mesh and model assets."""

from __future__ import annotations

from .formats.manifest import (
    MANIFEST_NAME,
    load_manifest,
)
from .formats.object_manifest import OBJECT_MANIFEST_SCHEMA
from .native import native_bin_dir, run_native

__all__ = [
    "MANIFEST_NAME",
    "OBJECT_MANIFEST_SCHEMA",
    "load_manifest",
    "native_bin_dir",
    "run_native",
]

__version__ = "0.4.0"
