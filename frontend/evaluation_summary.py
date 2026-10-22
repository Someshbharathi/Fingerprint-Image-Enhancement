"""Presentation helpers for summaries of persisted evaluation rows."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Mapping

import numpy as np


QUALITY_METRICS = ("psnr_db", "ssim", "mse")


def mean_quality_metrics(rows: Any) -> dict[str, float | None]:
    """Return arithmetic per-image means, excluding missing/invalid values.

    This mirrors the backend run summary aggregation: arithmetic means over
    finite per-image metric values. Latency and total runtime are never read.
    """
    means: dict[str, float | None] = {}
    valid_rows = tuple(rows) if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes, Mapping)) else ()
    for metric in QUALITY_METRICS:
        values: list[float] = []
        for row in valid_rows:
            if not isinstance(row, Mapping):
                continue
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
        means[metric] = float(np.mean(values)) if values else None
    return means


def format_quality_mean(value: float | None, metric: str) -> str:
    """Format a finite mean at the precision required by the UI."""
    precision = {"psnr_db": 2, "ssim": 4, "mse": 6}.get(metric)
    if precision is None:
        raise ValueError(f"Unsupported quality metric: {metric}")
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"
