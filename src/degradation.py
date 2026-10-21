import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def degrade_image(clean_image, seed=None, noise_std_range=(0.0, 0.025),
                  blur_probability=0.40, contrast_range=(0.65, 1.10),
                  brightness_range=(-0.015, 0.015)):
    """Make a reproducibly seeded synthetic corruption; does not modify input."""
    rng = np.random.default_rng(seed)
    image = clean_image.convert("L").copy()
    if rng.random() < 0.60:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(*contrast_range)))
    if rng.random() < 0.40:
        image = ImageEnhance.Brightness(image).enhance(1.0 + float(rng.uniform(*brightness_range)))
    if rng.random() < blur_probability:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7))))
    arr = np.asarray(image, dtype=np.float32).copy() / 255.0
    sigma = float(rng.uniform(*noise_std_range))
    if sigma:
        arr += rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray(np.round(arr * 255).astype(np.uint8), mode="L")
