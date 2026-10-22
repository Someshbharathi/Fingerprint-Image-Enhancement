
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FingerprintEnhancementDataset
from .model import ResNetEnhancer
from .utils import set_seed, get_device, read_manifest


def gradient_loss(pred, target):
    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = sobel_x.transpose(2, 3)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)

    return (
        F.l1_loss(pred_x, target_x)
        + F.l1_loss(pred_y, target_y)
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    epoch=1,
    total_epochs=1,
    phase="Train",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [{phase}]",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(inputs)

            l1 = criterion(predictions, targets)
            edge = gradient_loss(predictions, targets)
            loss = l1 + 0.1 * edge

            if training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        # Update the progress display periodically to reduce
        # repeated device-to-CPU synchronization.
        if (
            total_samples == batch_size
            or total_samples % (batch_size * 25) == 0
        ):
            average_loss = total_loss / total_samples
            progress.set_postfix(
                average=f"{average_loss:.5f}",
                l1=f"{l1.detach().item():.5f}",
                edge=f"{edge.detach().item():.5f}",
            )

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    validation_loss,
    best_validation_loss,
    image_size,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_l1": validation_loss,
        "best_validation_loss": best_validation_loss,
        "image_size": image_size,
        "features": 64,
        "num_blocks": 8,
    }

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train the ResNet fingerprint enhancer."
    )

    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--splits-dir",
        default="outputs/dataset_splits",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/checkpoints/last.pt",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick test using a few images only.",
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive.")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs requested: {args.epochs}")
    print("Training mode: full dataset" if not args.smoke_test
          else "Training mode: smoke test")

    split_dir = Path(args.splits_dir)

    train_paths = read_manifest(
        split_dir / "train.csv", "train"
    )
    val_paths = read_manifest(
        split_dir / "validation.csv", "validation"
    )

    if not train_paths or not val_paths:
        raise RuntimeError(
            "Training or validation split is empty. "
            "Check your dataset split CSV files."
        )

    # Limit the actual paths BEFORE creating datasets/loaders.
    if args.smoke_test:
        train_paths = train_paths[:4]
        val_paths = val_paths[:2]
        args.epochs = 1
        args.batch_size = 2
        print("Smoke test: 4 training images, 2 validation images.")

    train_dataset = FingerprintEnhancementDataset(
        paths=train_paths,
        image_size=args.image_size,
        training=True,
        seed=args.seed,
    )

    val_dataset = FingerprintEnhancementDataset(
        paths=val_paths,
        image_size=args.image_size,
        training=False,
        evaluation_degradation=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    model = ResNetEnhancer(
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=8,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )
    criterion = nn.L1Loss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    history_path = Path("outputs/metrics/training_history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_validation_loss = float("inf")

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {last_path}"
            )

        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("image_size", args.image_size) != args.image_size:
            raise ValueError(
                "Checkpoint image size differs from --image-size."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint.get(
            "best_validation_loss",
            checkpoint.get("val_l1", float("inf")),
        )

        print(f"Resuming from epoch {start_epoch}.")
        print(f"Best validation loss so far: {best_validation_loss:.6f}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}. "
            "Increase --epochs to continue training."
        )
        return

    # Keep previous history when resuming; create a fresh history
    # only for a new training run.
    if args.resume and history_path.exists():
        history_file = history_path.open(
            "a", newline="", encoding="utf-8"
        )
    else:
        history_file = history_path.open(
            "w", newline="", encoding="utf-8"
        )

    with history_file as file:
        writer = csv.writer(file)

        if not args.resume or file.tell() == 0:
            writer.writerow(
                ["epoch", "train_loss", "validation_loss"]
            )

        epoch_progress = tqdm(
            range(start_epoch, args.epochs + 1),
            desc="Overall training",
            unit="epoch",
        )

        for epoch in epoch_progress:
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                total_epochs=args.epochs,
                phase="Train",
            )

            validation_loss = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                epoch=epoch,
                total_epochs=args.epochs,
                phase="Validation",
            )

            writer.writerow(
                [epoch, train_loss, validation_loss]
            )
            file.flush()

            is_best = validation_loss < best_validation_loss

            if is_best:
                best_validation_loss = validation_loss

            save_checkpoint(
                last_path,
                epoch,
                model,
                optimizer,
                validation_loss,
                best_validation_loss,
                args.image_size,
            )

            if is_best:
                save_checkpoint(
                    best_path,
                    epoch,
                    model,
                    optimizer,
                    validation_loss,
                    best_validation_loss,
                    args.image_size,
                )

            epoch_progress.set_postfix(
                train=f"{train_loss:.5f}",
                val=f"{validation_loss:.5f}",
                best=f"{best_validation_loss:.5f}",
            )

            print(
                f"\nEpoch {epoch}/{args.epochs}"
                f" | Train Loss: {train_loss:.6f}"
                f" | Validation Loss: {validation_loss:.6f}"
                f" | Best Validation Loss: {best_validation_loss:.6f}"
            )

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {last_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
