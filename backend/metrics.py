import numpy as np
from skimage.metrics import structural_similarity

def calculate_metrics(prediction, reference):
    p = np.asarray(prediction, dtype=np.float64).squeeze()
    r = np.asarray(reference, dtype=np.float64).squeeze()
    if p.shape != r.shape or p.ndim != 2: raise ValueError("Prediction and reference must be matching 2D arrays")
    p = np.clip(p, 0, 1); r = np.clip(r, 0, 1)
    mse = float(np.mean((p-r)**2))
    # JSON has no representation for +infinity; exact reconstruction has undefined finite PSNR.
    psnr = None if mse == 0 else float(10*np.log10(1.0/mse))
    return {"mse": mse, "psnr_db": psnr, "ssim": float(structural_similarity(r, p, data_range=1.0))}
