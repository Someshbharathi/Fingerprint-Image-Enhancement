import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    try:
        import torch_directml

        device = torch_directml.device()

        # Verify that DirectML operations work
        test = torch.tensor([1.0]).to(device)
        _ = (test + 1).item()

        return device

    except Exception as e:
        print(f"DirectML unavailable: {e}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def read_manifest(csv_path, split_name):
    import csv
    from pathlib import Path
    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split_name and row.get("eligibility") in (
                "eligible_candidate_review", "eligible_candidate"
            ):
                rows.append(row["path"])
    if not rows:
        raise ValueError(f"No paths found for split={split_name} in {csv_path}")
    return rows
