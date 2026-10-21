from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from .preprocessing import IMAGE_EXTENSIONS, open_grayscale, resize_and_pad, image_to_tensor
from .degradation import degrade_image

class FingerprintEnhancementDataset(Dataset):
    """Synthetic paired data: degraded original -> original target."""
    def __init__(self, root=None, paths=None, image_size=256, training=True,
                 evaluation_degradation=False, seed=1234):
        self.image_size = int(image_size)
        self.training = bool(training)
        self.evaluation_degradation = bool(evaluation_degradation)
        self.seed = int(seed)
        if paths is not None:
            found = [Path(p) for p in paths]
        elif root is not None:
            root = Path(root).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {root}")
            found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        else:
            raise ValueError("Provide root or paths")
        self.paths = sorted(p for p in found if p.is_file())
        if not self.paths:
            raise ValueError("No supported images found")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        clean = resize_and_pad(open_grayscale(path), self.image_size)
        if self.training:
            degraded = degrade_image(clean, seed=random.getrandbits(32))
        elif self.evaluation_degradation:
            degraded = degrade_image(clean, seed=self.seed + index)
        else:
            degraded = clean.copy()
        return {"input": image_to_tensor(degraded, self.image_size),
                "target": image_to_tensor(clean, self.image_size),
                "path": str(path)}
