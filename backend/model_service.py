from __future__ import annotations
import io, logging, threading, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from src.model import ResNetEnhancer
from src.preprocessing import grayscale_image, tensor_to_image
from src.utils import get_device

log = logging.getLogger(__name__)
ALLOWED = {"png", "jpg", "jpeg", "tif", "tiff","bmp"}

class ModelService:
    def __init__(self, checkpoint: Path):
        self.path = Path(checkpoint)
        self.lock = threading.Lock()
        self.model = None; self.error = None; self.device = None
        self.image_size = None; self.info = {}
        self.latest_inference = None
        self._load()

    def _load(self):
        self.info = {"architecture": "ResNetEnhancer (src/model.py)", "checkpoint_path": str(self.path),
                     "checkpoint_exists": self.path.is_file(), "loaded": False,
                     "expected_size": None, "channels": 1, "input": "grayscale, aspect-preserving resize and white padding; float32 [0,1]",
                     "output_range": "[0,1]", "device": None}
        if not self.path.is_file():
            self.error = f"Checkpoint not found: {self.path}"
            return
        try:
            # Existing project checkpoint includes NumPy RNG metadata unsupported by weights_only=True.
            # This path is a configured local project artifact, never an uploaded request value.
            state = torch.load(self.path, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or not isinstance(state.get("model_state_dict"), dict):
                raise ValueError("Checkpoint must contain model_state_dict")
            self.image_size = int(state.get("image_size", 256))
            if self.image_size < 1: raise ValueError("Invalid image_size in checkpoint")
            model = ResNetEnhancer(1, 1, int(state.get("features", 64)), int(state.get("num_blocks", 8)))
            model.load_state_dict(state["model_state_dict"], strict=True)
            self.device = get_device()
            model.to(self.device).eval()
            self.model = model
            self.info.update({"loaded": True, "expected_size": [self.image_size, self.image_size],
                              "device": str(self.device), "epoch": state.get("epoch"),
                              "features": int(state.get("features",64)), "num_blocks": int(state.get("num_blocks",8))})
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("Model initialization failed")

    def status(self):
        return {**self.info, "error": self.error, "latest_inference": self.latest_inference,
                "ready": self.model is not None}

    @staticmethod
    def decode(source, filename: str | None = None):
        """Decode upload bytes, a binary stream, or a Pillow image to grayscale.

        A Pillow image is passed directly to preprocessing; it is never treated
        as a file stream. For TIFF and other multi-frame sources, frame zero is
        used consistently.
        """
        if isinstance(source, Image.Image):
            inferred_format = (source.format or "").lower()
            if filename is None:
                filename = {"tiff": "upload.tiff", "jpeg": "upload.jpg"}.get(inferred_format, f"upload.{inferred_format}" if inferred_format else "upload.png")
            if Path(filename).suffix.lower().lstrip(".") not in ALLOWED:
                raise ValueError("Unsupported image type. Use PNG, JPG, JPEG, TIF, or TIFF.")
            # Return directly through the decoded-image converter. Never route a
            # Pillow object through a file/stream opener such as Image.open().
            return grayscale_image(source)

        if isinstance(source, io.IOBase):
            data = source.read()
            if filename is None:
                filename = getattr(source, "name", None)
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            raise ValueError("Image input must be upload bytes, a binary stream, or a Pillow image")
        if not filename:
            raise ValueError("A filename is required to validate the image type")
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in ALLOWED:
            raise ValueError("Unsupported image type. Use PNG, JPG, JPEG, TIF, or TIFF.")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.seek(0)
                image.load()
                return grayscale_image(image)
        except (UnidentifiedImageError, OSError, EOFError) as exc:
            raise ValueError("The uploaded file is not a valid readable image") from exc

    def enhance(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError(self.error or "Model is unavailable")

        # Preserve the uploaded image dimensions.
        original = image.convert("L")
        original_size = original.size  # (width, height)

        size = self.image_size
        w, h = original_size

        # Resize proportionally to fit inside the model's square input.
        scale = min(size / w, size / h)
        resized_w = max(1, round(w * scale))
        resized_h = max(1, round(h * scale))

        resized = original.resize(
            (resized_w, resized_h),
            Image.Resampling.LANCZOS
        )

        # Center the resized image on the checkpoint's square input canvas.
        left = (size - resized_w) // 2
        top = (size - resized_h) // 2

        canvas = Image.new("L", (size, size), 255)
        canvas.paste(resized, (left, top))

        # Keep the checkpoint input as normalized (N,C,H,W) float32.
        x = torch.from_numpy(
            np.asarray(canvas, dtype=np.float32).copy() / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        started = time.perf_counter()

        with self.lock, torch.no_grad():
            y = self.model(x)

        elapsed = time.perf_counter() - started

        # Validate the model output.
        if y.ndim != 4 or tuple(y.shape) != (1, 1, size, size):
            raise RuntimeError(
                f"Unexpected model output shape: {tuple(y.shape)}"
            )

        # Convert prediction to a grayscale image.
        prediction = tensor_to_image(y[0])

        # Remove the white padding added before inference.
        prediction = prediction.crop((
            left,
            top,
            left + resized_w,
            top + resized_h
        ))

        # Restore the dimensions of the uploaded image.
        result = prediction.resize(
            original_size,
            Image.Resampling.LANCZOS
        )

        self.latest_inference = datetime.now(timezone.utc).isoformat()

        return result, elapsed, list(original_size)
