import argparse
from pathlib import Path
import torch
from .model import ResNetEnhancer
from .utils import get_device
from .preprocessing import open_grayscale, image_to_tensor, tensor_to_image

def main():
    p=argparse.ArgumentParser(description="Enhance one fingerprint image.")
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="outputs/enhanced/enhanced.png")
    p.add_argument("--checkpoint",default="outputs/checkpoints/best.pt")
    a=p.parse_args()
    device=get_device()
    state=torch.load(a.checkpoint,map_location=device)
    model=ResNetEnhancer(features=state.get("features",64), num_blocks=state.get("num_blocks",8)).to(device)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    size=state.get("image_size",256)
    x=image_to_tensor(open_grayscale(a.input),size).unsqueeze(0).to(device)
    with torch.no_grad(): pred=model(x)[0]
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    tensor_to_image(pred).save(output)
    print("Device:",device)
    print("Enhanced image saved:",output.resolve())
    print("Note: output is resized/padded to the model's square input size.")

if __name__=="__main__": main()
