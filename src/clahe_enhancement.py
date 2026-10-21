
# Generate U-Net outputs and save them for CLAHE
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

ENHANCED_DIR = OUTPUT_DIR / "enhanced_images"
ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

model.eval()
saved_count = 0

with torch.no_grad():
    for index, (inputs, targets) in enumerate(
        tqdm(test_loader, desc="Saving enhanced fingerprints")
    ):
        inputs = inputs.to(DEVICE)
        predictions = model(inputs).detach().cpu().numpy()[:, 0]

        for pred in predictions:
            # Convert model output from [0, 1] to grayscale [0, 255]
            image = np.clip(pred * 255, 0, 255).astype(np.uint8)

            output_path = ENHANCED_DIR / f"fingerprint_{saved_count:05d}.png"
            if cv2.imwrite(str(output_path), image):
                saved_count += 1

print(f"Saved {saved_count} images to: {ENHANCED_DIR.resolve()}")
