import argparse, csv, json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from .preprocessing import IMAGE_EXTENSIONS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True)
    p.add_argument("--report-dir",default="outputs/metrics/data_validation")
    a=p.parse_args()
    root=Path(a.data_root).expanduser().resolve()
    if not root.is_dir(): raise SystemExit(f"Dataset root not found: {root}")
    paths=sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
    rows=[]; bad=[]
    for i,path in enumerate(paths,1):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w,h=im.size; mode=im.mode
            rows.append({"relative_path":path.relative_to(root).as_posix(),"width":w,"height":h,"mode":mode,"status":"ok","error":""})
        except (OSError,ValueError,UnidentifiedImageError) as e:
            row={"relative_path":path.relative_to(root).as_posix(),"width":"","height":"","mode":"","status":"unreadable","error":str(e)}
            rows.append(row); bad.append(row)
        if i%5000==0: print(f"Checked {i:,}/{len(paths):,}")
    out=Path(a.report_dir); out.mkdir(parents=True,exist_ok=True)
    for name, data in [("image_inventory.csv",rows),("unreadable_images.csv",bad)]:
        with (out/name).open("w",newline="",encoding="utf-8-sig") as f:
            fields=list(data[0]) if data else ["relative_path","width","height","mode","status","error"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    summary={"dataset_root":str(root),"images_discovered":len(paths),"readable_images":sum(r["status"]=="ok" for r in rows),"unreadable_images":len(bad)}
    (out/"dataset_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print("Reports:",out.resolve())
if __name__=="__main__": main()
