"""Small, testable visualizations used by the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

QUALITY_METRICS = ("psnr_db", "ssim", "mse")


def pixel_intensity_histogram(original: np.ndarray, enhanced: np.ndarray) -> go.Figure:
    """Compare pixel-value distributions without assuming images share a size.

    The model can emit its checkpoint's fixed dimensions while the source image
    keeps its uploaded dimensions. Histograms are independent distributions;
    pixels are not paired observations and must not be put in equal-length
    dataframe columns.
    """
    original_values = np.asarray(original).ravel()
    enhanced_values = np.asarray(enhanced).ravel()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=original_values, name="Original", nbinsx=64, opacity=0.65))
    fig.add_trace(go.Histogram(x=enhanced_values, name="Enhanced", nbinsx=64, opacity=0.65))
    fig.update_layout(barmode="overlay", xaxis_title="Intensity (0–255)", yaxis_title="Pixel count")
    return fig


def identical_image_quality_metrics(runs: list[dict]) -> bool:
    """Whether selected runs have the same ordered per-image quality values.

    Inference latency is intentionally excluded because it varies between runs
    and is presented in its own comparison chart.
    """
    if len(runs) < 2:
        return False

    def signature(run: dict) -> list[tuple]:
        return [
            (row.get("image"), *(row.get(metric) for metric in QUALITY_METRICS))
            for row in run.get("rows", [])
        ]

    reference = signature(runs[0])
    return all(signature(run) == reference for run in runs[1:])


def evaluation_metric_histogram(runs: list[dict], metric: str, bins: int = 30) -> go.Figure:
    """Build run-grouped overlays using one explicit set of bin edges."""
    if metric not in (*QUALITY_METRICS, "inference_ms"):
        raise ValueError(f"Unsupported evaluation metric: {metric}")
    if bins < 1:
        raise ValueError("bins must be positive")

    values_by_run: list[tuple[str, np.ndarray]] = []
    pooled = []
    for run in runs:
        run_id = str(run.get("run_id", "unknown run"))
        raw = [row.get(metric) for row in run.get("rows", [])]
        values = np.asarray([v for v in raw if v is not None and np.isfinite(v)], dtype=np.float64)
        if values.size:
            values_by_run.append((run_id, values))
            pooled.append(values)

    fig = go.Figure()
    if pooled:
        edges = np.histogram_bin_edges(np.concatenate(pooled), bins=bins)
        common_bins = {"start": float(edges[0]), "end": float(edges[-1]), "size": float(edges[1] - edges[0])}
        for run_id, values in values_by_run:
            fig.add_trace(go.Histogram(
                x=values,
                name=run_id,
                opacity=0.48,
                histnorm="probability",
                xbins=common_bins,
            ))
    fig.update_layout(barmode="overlay", xaxis_title=metric, yaxis_title="Share of images")
    return fig
