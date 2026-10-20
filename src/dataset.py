
from pathlib import Path
import cv2

# Locate the project root (the parent folder of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"

DATASETS = ["FVC2000", "FVC2002", "FVC2004", "SOCOFing"]

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def inspect_dataset(name):
    folder = DATASET_ROOT / name

    print(f"\n{'=' * 50}")
    print(f"DATASET: {name}")
    print(f"Path: {folder}")
    print(f"{'=' * 50}")

    if not folder.exists():
        print("ERROR: Dataset folder not found!")
        return

    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("Total images found:", len(image_paths))

    if not image_paths:
        print("No supported images found. Check the folder structure.")
        return

    print("\nFirst 5 image paths:")
    for path in image_paths[:5]:
        print(" ", path.relative_to(PROJECT_ROOT))

    print("\nInspecting image dimensions:")
    readable = 0
    unreadable = 0
    dimensions = {}

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable += 1
            continue

        readable += 1
        height, width = image.shape
        dimensions[(height, width)] = dimensions.get(
            (height, width), 0
        ) + 1

    print("Readable images:", readable)
    print("Unreadable images:", unreadable)
    print("Image dimensions (height, width):")

    for shape, count in sorted(
        dimensions.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"  {shape}: {count} images")


if __name__ == "__main__":
    for dataset_name in DATASETS:
        inspect_dataset(dataset_name)
