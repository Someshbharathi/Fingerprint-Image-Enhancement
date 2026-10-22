from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".bmp", ".dib", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

def grayscale_image(im: Image.Image):
    """Convert an already-decoded Pillow image to an owned grayscale copy.

    Multi-frame formats use their first frame. This function never opens a path
    or treats the Pillow image as a file-like stream.
    """
    try:
        im.seek(0)
        im.load()
    except (EOFError, OSError) as exc:
        raise ValueError("The image's first frame could not be decoded") from exc
    if not im.width or not im.height:
        raise ValueError("Image dimensions are invalid")
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return im.convert("L").copy()

def open_grayscale(path):
    """Read an image path or convert a decoded Pillow image to grayscale."""
    if isinstance(path, Image.Image):
        return grayscale_image(path)
    with Image.open(path) as im:
        return grayscale_image(im)

def resize_and_pad(image, size=256, pad_value=255):
    if size <= 0:
        raise ValueError("size must be positive")
    w, h = image.size
    if not w or not h:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), pad_value)
    canvas.paste(image, ((size - nw)//2, (size - nh)//2))
    return canvas

def image_to_tensor(image, size=256):
    image = resize_and_pad(image.convert("L"), size)
    arr = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_image(tensor):
    arr = tensor.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    return Image.fromarray(np.round(arr * 255).astype(np.uint8), mode="L")
