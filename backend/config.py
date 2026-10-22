from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    checkpoint: Path = Path(os.getenv("FPRINT_CHECKPOINT", ROOT / "outputs/checkpoints/best.pt"))
    test_manifest: Path = Path(os.getenv("FPRINT_TEST_MANIFEST", ROOT / "outputs/dataset_splits/test.csv"))
    results_dir: Path = Path(os.getenv("FPRINT_RESULTS_DIR", ROOT / "outputs/dashboard"))
    max_upload_mb: int = int(os.getenv("FPRINT_MAX_UPLOAD_MB", "12"))
    host: str = os.getenv("FPRINT_API_HOST", "127.0.0.1")
    port: int = int(os.getenv("FPRINT_API_PORT", "5000"))

settings = Settings()
