
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FingerprintEnhancementDataset
from .model import ResNetEnhancer
from .utils import get_device, read_manifest
from .preprocessing import tensor_to_image


def metrics(pred, target):
    """Calculate image-quality metrics for one image."""
    p = pred.detach().cpu().numpy().clip(0, 1).squeeze()
    t = target.detach().cpu().numpy().clip(0, 1).squeeze()

    mse = float(np.mean((p - t) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))

    try:
        from skimage.metrics import structural_similarity

        ssim = float(
            structural_similarity(
                t,
                p,
                data_range=1.0
            )
        )
    except ImportError:
        ssim = float("nan")

    rmse = float(np.sqrt(mse))

    return mse, rmse, psnr, ssim


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fingerprint enhancement model."
    )

    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--splits-dir",
        default="outputs/dataset_splits"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/comparisons/test"
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    device = get_device()
    print("Device:", device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = ResNetEnhancer(
        in_channels=1,
        out_channels=1,
        features=checkpoint.get("features", 64),
        num_blocks=checkpoint.get("num_blocks", 8)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Checkpoint validation L1:", checkpoint.get("val_l1"))

    test_manifest = Path(args.splits_dir) / "test.csv"
    test_paths = read_manifest(test_manifest, "test")

    if not test_paths:
        raise RuntimeError("The test manifest contains no images.")

    dataset = FingerprintEnhancementDataset(
        paths=test_paths,
        image_size=checkpoint.get("image_size", 256),
        training=False,
        evaluation_degradation=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path("outputs/metrics/test_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating test set"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)

            for i, image_path in enumerate(batch["path"]):
                mse, rmse, psnr, ssim = metrics(
                    predictions[i],
                    targets[i]
                )

                stem = Path(image_path).stem

                tensor_to_image(inputs[i]).save(
                    output_dir / f"{stem}_degraded.png"
                )

                tensor_to_image(predictions[i]).save(
                    output_dir / f"{stem}_enhanced.png"
                )

                tensor_to_image(targets[i]).save(
                    output_dir / f"{stem}_target.png"
                )

                rows.append({
                    "path": image_path,
                    "mse": mse,
                    "rmse": rmse,
                    "psnr_db": psnr,
                    "ssim": ssim
                })

    fieldnames = [
        "path",
        "mse",
        "rmse",
        "psnr_db",
        "ssim"
    ]

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print("\nTest-set results")
        print("Images evaluated:", len(rows))
        print("Mean MSE:", np.mean([r["mse"] for r in rows]))
        print("Mean RMSE:", np.mean([r["rmse"] for r in rows]))
        print("Mean PSNR (dB):", np.mean([r["psnr_db"] for r in rows]))
        print("Mean SSIM:", np.nanmean([r["ssim"] for r in rows]))

    print("\nMetrics saved to:", metrics_path)
    print("Comparison images saved to:", output_dir)


if __name__ == "__main__":
    main()
