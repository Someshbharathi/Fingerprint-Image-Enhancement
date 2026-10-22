import numpy as np

from frontend.visualization import (
    evaluation_metric_histogram,
    identical_image_quality_metrics,
    pixel_intensity_histogram,
)


def test_pixel_histogram_accepts_model_output_with_different_dimensions():
    # Typical source and checkpoint output dimensions from the dashboard flow.
    original = np.zeros((300, 300), dtype=np.uint8)
    enhanced = np.full((256, 256), 127, dtype=np.uint8)

    fig = pixel_intensity_histogram(original, enhanced)

    assert len(fig.data) == 2
    assert fig.data[0].name == "Original"
    assert len(fig.data[0].x) == 300 * 300
    assert fig.data[1].name == "Enhanced"
    assert len(fig.data[1].x) == 256 * 256
    assert fig.layout.barmode == "overlay"


def test_pixel_histogram_flattens_single_channel_images():
    fig = pixel_intensity_histogram(np.zeros((2, 3, 1)), np.ones((4, 2, 1)))
    assert len(fig.data[0].x) == 6
    assert len(fig.data[1].x) == 8


def test_identical_quality_metrics_keep_identical_histograms_when_latency_differs():
    rows = [
        {
            "image": f"finger_{index:04d}.tif",
            "psnr_db": 20.0 + (index % 37) / 10,
            "ssim": 0.7 + (index % 19) / 100,
            "mse": (index % 23) / 1000,
            "inference_ms": 3.0 + (index % 11) / 10,
        }
        for index in range(1560)
    ]
    repeated_rows = [{**row, "inference_ms": row["inference_ms"] + 4.0} for row in rows]
    runs = [
        {"run_id": "evaluation-run-one-full-id", "rows": rows},
        {"run_id": "evaluation-run-two-full-id", "rows": repeated_rows},
    ]

    assert identical_image_quality_metrics(runs)
    for metric in ("psnr_db", "ssim", "mse"):
        fig = evaluation_metric_histogram(runs, metric)
        assert [trace.name for trace in fig.data] == [run["run_id"] for run in runs]
        assert np.array_equal(np.asarray(fig.data[0].x), np.asarray(fig.data[1].x))
        assert fig.data[0].xbins.to_plotly_json() == fig.data[1].xbins.to_plotly_json()

    latency_fig = evaluation_metric_histogram(runs, "inference_ms")
    assert not np.array_equal(np.asarray(latency_fig.data[0].x), np.asarray(latency_fig.data[1].x))
    assert latency_fig.data[0].xbins.to_plotly_json() == latency_fig.data[1].xbins.to_plotly_json()
