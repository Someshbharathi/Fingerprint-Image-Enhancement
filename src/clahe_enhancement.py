
from pathlib import Path
import cv2

# Project root: Fingerprint-Image-Enhancement
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input: U-Net-generated images
ENHANCED_DIR = PROJECT_DIR / "outputs" / "enhanced_images"

# Output: CLAHE-processed images
CLAHE_DIR = PROJECT_DIR / "outputs" / "enhanced_images_clahe"

def main():
    if not ENHANCED_DIR.exists():
        print("U-Net output folder does not exist:")
        print(ENHANCED_DIR)
        print("Generate U-Net predictions first.")
        return

    image_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }

    image_paths = [
        p for p in ENHANCED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print("No images found in:", ENHANCED_DIR)
        print("Generate U-Net predictions first.")
        return

    CLAHE_DIR.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    saved_count = 0

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Could not read:", path.name)
            continue

        output = clahe.apply(image)
        output_path = CLAHE_DIR / path.name

        if cv2.imwrite(str(output_path), output):
            saved_count += 1

    print(f"CLAHE completed: {saved_count} images saved.")
    print("Input folder:", ENHANCED_DIR)
    print("Output folder:", CLAHE_DIR)

if __name__ == "__main__":
    main()
