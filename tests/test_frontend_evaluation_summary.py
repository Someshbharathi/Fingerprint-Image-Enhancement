import pytest

from frontend.evaluation_summary import format_quality_mean, mean_group_metrics, mean_quality_metrics, metric_change


def test_summary_uses_arithmetic_means_of_selected_run_rows_only():
    rows = [
        {"psnr_db": 20.0, "ssim": 0.8, "mse": 0.001, "inference_ms": 7.0},
        {"psnr_db": 22.0, "ssim": 1.0, "mse": 0.003, "inference_ms": 900.0},
    ]

    summary = mean_quality_metrics(rows)

    assert summary == {"psnr_db": 21.0, "ssim": 0.9, "mse": 0.002}
    assert format_quality_mean(summary["psnr_db"], "psnr_db") == "21.00"
    assert format_quality_mean(summary["ssim"], "ssim") == "0.9000"
    assert format_quality_mean(summary["mse"], "mse") == "0.002000"


def test_summary_returns_na_for_empty_or_invalid_metric_values():
    summary = mean_quality_metrics([
        {"psnr_db": None, "ssim": float("nan"), "mse": "invalid"},
        {"psnr_db": float("inf"), "ssim": True},
        None,
    ])

    assert summary == {"psnr_db": None, "ssim": None, "mse": None}
    assert all(format_quality_mean(value, metric) == "N/A" for metric, value in summary.items())
    assert format_quality_mean(None, "mse") == "N/A"


def test_summary_accepts_incomplete_rows_without_zero_filling():
    summary = mean_quality_metrics([{"psnr_db": 24.125}, {"mse": 0.000002}])

    assert summary == {"psnr_db": 24.125, "ssim": None, "mse": 0.000002}


def test_group_summaries_and_change_are_separate():
    rows = [
        {"baseline": {"psnr_db": 10.0, "ssim": 0.5, "mse": 0.1},
         "enhanced": {"psnr_db": 20.0, "ssim": 0.8, "mse": 0.01}},
        {"baseline": {"psnr_db": 14.0, "ssim": 0.7, "mse": 0.03},
         "enhanced": {"psnr_db": 22.0, "ssim": 0.9, "mse": 0.005}},
    ]
    baseline = mean_group_metrics(rows, "baseline")
    enhanced = mean_group_metrics(rows, "enhanced")
    assert baseline == {"psnr_db": 12.0, "ssim": 0.6, "mse": 0.065}
    assert enhanced["psnr_db"] == 21.0
    assert enhanced["ssim"] == pytest.approx(0.85)
    assert enhanced["mse"] == pytest.approx(0.0075)
    assert metric_change(baseline["psnr_db"], enhanced["psnr_db"], "psnr_db") == 9.0
    assert metric_change(baseline["mse"], enhanced["mse"], "mse") == -0.0575
    assert metric_change(None, enhanced["psnr_db"], "psnr_db") is None
