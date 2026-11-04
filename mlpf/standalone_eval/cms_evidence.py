"""Paths shared by CMS qualification runners and report tools."""

from pathlib import Path


def input_manifest_path(root):
    """Prefer prepared inputs; retain support for archived evidence packages."""
    root = Path(root)
    prepared = root / "inputs/manifest.json"
    archived = root / "input-manifest.json"
    return archived if not prepared.exists() and archived.exists() else prepared
