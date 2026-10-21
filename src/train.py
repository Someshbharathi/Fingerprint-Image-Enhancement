import argparse, csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import FingerprintEnhancementDataset
from .model import ResNetEnhancer
from .utils import set_seed, get_device, read_manifest
import torch.nn.functional as F


def gradient_loss(pred, target):
    # Sobel filters measure horizontal and vertical image gradients.
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
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


def run_epoch(model, loader, criterion, device, optimizer=None, epoch=1, total_epochs=1, phase="Train"):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [{phase}]",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in progress:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
            l1 = criterion(pred, y)
            edge = gradient_loss(pred, y)
            loss = l1 + 0.1 * edge

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        avg_loss = total_loss / total_samples

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
            l1=f"{l1.item():.4f}",
            edge=f"{edge.item():.4f}",
        )

    return total_loss / max(total_samples, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Dataset root")
    p.add_argument("--splits-dir", default="outputs/dataset_splits")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()

    set_seed(a.seed)
    device = get_device()
    print(f"Device: {device}")
    split_dir = Path(a.splits_dir)
    train_paths = read_manifest(split_dir / "train.csv", "train")
    val_paths = read_manifest(split_dir / "validation.csv", "validation")

    train_ds = FingerprintEnhancementDataset(
        paths=train_paths,
        image_size=a.image_size,
        training=True,
        seed=a.seed
    )

    val_ds = FingerprintEnhancementDataset(
        paths=val_paths,
        image_size=a.image_size,
        training=False,
        evaluation_degradation=True,
        seed=a.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda")
    )
    model = ResNetEnhancer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    criterion = nn.L1Loss()
    ckpt_dir = Path(a.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history_path = Path("outputs/metrics/training_history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if a.smoke_test:
        a.epochs = 1
        a.batch_size = 2
        train_paths = train_paths[:4]
        val_paths = val_paths[:2]
        print("SMOKE TEST: using 4 training images and 2 validation images.")
        
    
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "validation_loss"])

        epoch_bar = tqdm(
            range(1, a.epochs + 1),
            desc="Training progress",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            tr = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer,
                epoch=epoch,
                total_epochs=a.epochs,
                phase="Train",
            )

            va = run_epoch(
                model, val_loader, criterion, device,
                epoch=epoch,
                total_epochs=a.epochs,
                phase="Validation",
            )

            writer.writerow([epoch, tr, va])
            f.flush()

            epoch_bar.set_postfix(
                train=f"{tr:.4f}",
                val=f"{va:.4f}",
                best=f"{min(best, va):.4f}",
            )

            print(
                f"\nEpoch {epoch}/{a.epochs} | "
                f"Train Loss: {tr:.6f} | "
                f"Validation Loss: {va:.6f}"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1": va,
                "image_size": a.image_size,
                "features": 64,
                "num_blocks": 8,
            }

            torch.save(state, ckpt_dir / "last.pt")

            if va < best:
                best = va
                torch.save(state, ckpt_dir / "best.pt")
                
    print(f"Best validation L1: {best:.6f}")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
