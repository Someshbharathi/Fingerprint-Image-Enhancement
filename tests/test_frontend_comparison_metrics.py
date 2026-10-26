import io

import numpy as np
import pytest
from PIL import Image

from backend.metrics import calculate_metrics
from frontend.comparison_metrics import calculate_comparison_metrics, calculate_input_similarity_metrics


def png_bytes(array, mode="L"):
    output = io.BytesIO()
    Image.fromarray(array, mode=mode).save(output, format="PNG")
    return output.getvalue()


def test_no_reference_returns_no_quality_metrics():
    enhanced = png_bytes(np.full((16, 16), 100, dtype=np.uint8))

    assert calculate_comparison_metrics(enhanced, None, (16, 16)) is None


def test_reference_metrics_match_backend_metrics_and_include_rmse():
    reference = np.full((16, 16), 128, dtype=np.uint8)
    enhanced = reference.copy()
    enhanced[4, 7] = 160
    expected = calculate_metrics(enhanced.astype(np.float32) / 255.0,
                                 reference.astype(np.float32) / 255.0)

    result = calculate_comparison_metrics(png_bytes(enhanced), png_bytes(reference), (16, 16))

    assert result is not None
    assert result["mse"] == expected["mse"]
    assert result["psnr_db"] == expected["psnr_db"]
    assert result["ssim"] == expected["ssim"]
    assert result["rmse"] == pytest.approx(np.sqrt(expected["mse"]))


def test_reference_dimensions_must_match_exactly():
    enhanced = png_bytes(np.zeros((16, 16), dtype=np.uint8))
    reference = png_bytes(np.zeros((16, 15), dtype=np.uint8))

    with pytest.raises(ValueError, match="must match"):
        calculate_comparison_metrics(enhanced, reference, (16, 16))


def test_metric_comparison_rejects_output_not_at_input_dimensions():
    enhanced = png_bytes(np.zeros((15, 16), dtype=np.uint8))
    reference = png_bytes(np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected the uploaded input size"):
        calculate_comparison_metrics(enhanced, reference, (16, 16))


def test_input_output_similarity_is_calculated_and_kept_separate():
    image = png_bytes(np.full((16, 16), 128, dtype=np.uint8))

    result = calculate_input_similarity_metrics(image, image, (16, 16))

    assert result == {"mse": 0.0, "psnr_db": None, "ssim": 1.0, "rmse": 0.0}
