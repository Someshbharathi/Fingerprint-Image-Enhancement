"""State transitions for the image enhancement workflow."""

from __future__ import annotations

from typing import Any, MutableMapping


def select_upload(state: MutableMapping[str, Any], data: bytes, filename: str, size: tuple[int, int]) -> None:
    """Store an upload and discard a result that belongs to another image."""
    current = state.get("upload")
    if not current or current.get("data") != data or current.get("filename") != filename:
        state["result"] = None
    state["upload"] = {"data": data, "filename": filename, "size": size}


def clear_upload(state: MutableMapping[str, Any]) -> None:
    """Clear stale input and output after the user selects an invalid file."""
    state["upload"] = None
    state["result"] = None
