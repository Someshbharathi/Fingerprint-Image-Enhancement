"""State transitions for the image enhancement workflow."""

from __future__ import annotations

import hashlib
from typing import Any, MutableMapping


def _upload_id(data: bytes, filename: str) -> str:
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8"))
    digest.update(b"\0")
    digest.update(data)
    return digest.hexdigest()


def select_upload(state: MutableMapping[str, Any], data: bytes, filename: str, size: tuple[int, int]) -> None:
    """Store an upload and invalidate output/reference state when it changes."""
    current = state.get("upload")
    if not current or current.get("data") != data or current.get("filename") != filename:
        state["result"] = None
        state["comparison_reference_data"] = None
    state["upload"] = {"data": data, "filename": filename, "size": size,
                       "session_id": _upload_id(data, filename)}


def store_result(state: MutableMapping[str, Any], data: bytes, meta: dict[str, Any]) -> None:
    """Bind an enhanced output to the exact upload session that produced it."""
    upload = state.get("upload")
    if not upload:
        raise ValueError("Cannot store an enhancement result without an uploaded image")
    state["result"] = {"data": data, "meta": meta, "session_id": upload["session_id"]}


def current_result(state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Return the result only when it belongs to the currently selected upload."""
    upload, result = state.get("upload"), state.get("result")
    if not upload or not result or result.get("session_id") != upload.get("session_id"):
        return None
    return result


def clear_upload(state: MutableMapping[str, Any]) -> None:
    """Clear stale input and output after the user selects an invalid file."""
    state["upload"] = None
    state["result"] = None
    state["comparison_reference_data"] = None
