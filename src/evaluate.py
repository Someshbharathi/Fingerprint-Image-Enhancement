import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import FingerprintEnhancementDataset
from .model import ResNetEnhancer
from .utils import get_device, read_manifest
from .preprocessing import tensor_to_image

def metrics(pred, target):
    p = pred.detach().cpu().numpy().clip(0,1)
    t = target.detach().cpu().numpy().clip(0,1)
    mse = float(np.mean((p-t)**2))
    psnr = float(10*np.log10(1.0/max(mse, 1e-12)))
    try:
        from skimage.metrics import structural_similarity
        ssims = [structural_similarity(ti[0], pi[0], data_range=1.0) for ti,pi in zip(t,p)]
        ssim = float(np.mean(ssims))
    except ImportError:
        ssim = float("nan")
    return mse, psnr, ssim

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--splits-dir", default="outputs/dataset_splits")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/comparisons/test")
    a=p.parse_args()
    device=get_device(); print("Device:",device)
    state=torch.load(a.checkpoint,map_location=device)
    model=ResNetEnhancer(features=state.get("features",64), num_blocks=state.get("num_blocks",8)).to(device)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    paths=read_manifest(Path(a.splits_dir)/"test.csv","test")
    ds=FingerprintEnhancementDataset(paths=paths,image_size=state.get("image_size",256),training=False,evaluation_degradation=True)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.no_grad():
        for batch in tqdm(loader):
            x,y=batch["input"].to(device),batch["target"].to(device)
            pred=model(x)
            mse,psnr,ssim=metrics(pred,y)
            for i,path in enumerate(batch["path"]):
                stem=Path(path).stem
                tensor_to_image(x[i]).save(out/f"{stem}_degraded.png")
                tensor_to_image(pred[i]).save(out/f"{stem}_enhanced.png")
                tensor_to_image(y[i]).save(out/f"{stem}_target.png")
                rows.append({"path":path,"mse":mse,"psnr_db":psnr,"ssim":ssim})
    csvpath=Path("outputs/metrics/test_metrics.csv"); csvpath.parent.mkdir(parents=True,exist_ok=True)
    with csvpath.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","mse","psnr_db","ssim"]); w.writeheader(); w.writerows(rows)
    if rows:
        print("Mean MSE:",np.mean([r["mse"] for r in rows]))
        print("Mean PSNR dB:",np.mean([r["psnr_db"] for r in rows]))
        print("Mean SSIM:",np.mean([r["ssim"] for r in rows]))
    print("Metrics:",csvpath,"| Comparisons:",out)

if __name__=="__main__": main()
