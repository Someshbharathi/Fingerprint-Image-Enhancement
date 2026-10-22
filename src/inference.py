
import argparse
from pathlib import Path

import torch

from .model import ResNetEnhancer
from .utils import get_device
from .preprocessing import (
    open_grayscale,
    image_to_tensor,
    tensor_to_image,
)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance one fingerprint image."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="outputs/enhanced/enhanced.png"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input fingerprint image not found: {input_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load checkpoint on CPU to avoid DirectML deserialization errors.
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    device = get_device()

    model = ResNetEnhancer(
        in_channels=1,
        out_channels=1,
        features=state.get("features", 64),
        num_blocks=state.get("num_blocks", 8)
    )

    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_size = state.get("image_size", 256)

    image = open_grayscale(input_path)
    input_tensor = image_to_tensor(
        image,
        image_size
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)[0]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tensor_to_image(prediction).save(output_path)

    print("Device:", device)
    print("Checkpoint epoch:", state.get("epoch", "unknown"))
    print("Input image:", input_path.resolve())
    print("Enhanced image saved:", output_path.resolve())
    print(
        f"Note: output is resized/padded to "
        f"{image_size}x{image_size}."
    )


if __name__ == "__main__":
    main()
