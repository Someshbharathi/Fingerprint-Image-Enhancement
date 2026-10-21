from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

# Project root = parent of this script's src folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "inputs_to_enhance"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "enhanced_images"

# Prefer checkpoint inside outputs/models; fall back to outputs root.
CHECKPOINT_CANDIDATES = [
    PROJECT_DIR / "outputs" / "models" / "unet_fingerprint_best_directml.pth",
    PROJECT_DIR / "outputs" / "unet_fingerprint_best_directml.pth",
    PROJECT_DIR / "unet_fingerprint_best_directml.pth",
]
IMAGE_SIZE = 256
BASE_CHANNELS_FALLBACK = 16
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.output(d1))


def main():
    checkpoint_path = next((p for p in CHECKPOINT_CANDIDATES if p.is_file()), None)
    if checkpoint_path is None:
        print("Could not find unet_fingerprint_best_directml.pth.")
        print("Place it in: ", CHECKPOINT_CANDIDATES[0])
        return

    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Created input folder:", INPUT_DIR)
        print("Copy 1 or more fingerprint images into this folder, then run this script again.")
        return

    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print("No images found in:", INPUT_DIR)
        print("Copy a few fingerprint images into this folder and run again.")
        return

    try:
        import torch_directml
        device = torch_directml.device()
    except Exception as exc:
        print("Could not initialize DirectML:", exc)
        print("Install/activate the same .venv-gpu environment used for training.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("base_channels", BASE_CHANNELS_FALLBACK))
    model = UNet(in_channels=1, out_channels=1, base=base_channels)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Skipping unreadable image:", path.name)
                continue
            resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
            prediction = model(tensor).detach().cpu().numpy()[0, 0]
            output = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
            out_path = OUTPUT_DIR / f"{path.stem}_unet.png"
            if cv2.imwrite(str(out_path), output):
                saved += 1
                print("Saved:", out_path.name)

    print(f"\nFinished. Saved {saved} U-Net output(s) to:")
    print(OUTPUT_DIR)
    print("Next, run src/clahe_enhancement.py to create CLAHE versions.")


if __name__ == "__main__":
    main()
