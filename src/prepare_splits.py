"""Create reproducible manifests. Grouping is conservative, not proof of identity separation."""
import argparse,csv,random,re
from pathlib import Path
from collections import defaultdict
from .preprocessing import IMAGE_EXTENSIONS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output-dir",default="outputs/dataset_splits")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--train-ratio",type=float,default=.8)
    p.add_argument("--val-ratio",type=float,default=.1)
    p.add_argument("--test-ratio",type=float,default=.1)
    a=p.parse_args()
    if min(a.train_ratio,a.val_ratio,a.test_ratio)<=0 or abs(a.train_ratio+a.val_ratio+a.test_ratio-1)>1e-6:
        raise SystemExit("Ratios must be positive and sum to 1.")
    root=Path(a.root).expanduser().resolve()
    if not root.is_dir(): raise SystemExit(f"Dataset root not found: {root}")
    images=sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
    families=defaultdict(lambda:defaultdict(list)); inventory=[]
    for path in images:
        rel=path.relative_to(root).as_posix(); parts=[x.lower() for x in Path(rel).parts]
        if parts[0].startswith("socofing") and "altered" in parts:
            dataset="SOCOFing"; role="altered"; eligibility="excluded_clean_target"
            group="altered::"+str(Path(rel).parent)
        elif parts[0].startswith("socofing"):
            dataset="SOCOFing"; role="real_candidate"; eligibility="eligible_candidate_review"
            m=re.match(r"^(\d+)",path.stem)
            group="socofing_id_"+m.group(1) if m else "unparsed::"+rel
        elif parts[0].startswith("fvc"):
            dataset=parts[0].upper(); role="fvc_candidate"; eligibility="eligible_candidate_review"
            group="fvc_partition::"+str(Path(rel).parent)
        else:
            dataset=parts[0]; role="unknown"; eligibility="review_required"; group="unverified::"+rel
        row={"path":str(path.resolve()),"relative_path":rel,"dataset":dataset,"role":role,"eligibility":eligibility,"group_id":group,"group_policy":"filename_id_heuristic" if dataset=="SOCOFing" and role=="real_candidate" else ("partition_not_identity_verified" if role=="fvc_candidate" else "excluded_or_unverified"),"split":""}
        inventory.append(row)
        if eligibility=="eligible_candidate_review": families[dataset][group].append(row)
    rng=random.Random(a.seed)
    for family,groups in families.items():
        keys=list(groups); rng.shuffle(keys)
        n=len(keys)
        if n<3:
            for rows in groups.values():
                for row in rows: row["eligibility"]="review_required_too_few_groups"
            continue
        counts=[int(n*a.train_ratio),int(n*a.val_ratio),int(n*a.test_ratio)]
        for i in sorted(range(3),key=lambda j:n*[a.train_ratio,a.val_ratio,a.test_ratio][j]-counts[j],reverse=True)[:n-sum(counts)]: counts[i]+=1
        # Guarantee each split a group when there are at least 3 groups.
        for i in range(3):
            if counts[i]==0:
                donor=max(range(3),key=lambda j:counts[j]); counts[donor]-=1; counts[i]+=1
        cuts=[counts[0],counts[0]+counts[1]]
        for idx,key in enumerate(keys):
            split="train" if idx<cuts[0] else ("validation" if idx<cuts[1] else "test")
            for row in groups[key]: row["split"]=split
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fields=["path","relative_path","dataset","role","eligibility","group_id","group_policy","split"]
    def write(name,rows):
        with (out/name).open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write("dataset_inventory.csv",inventory)
    for split in ("train","validation","test"):
        write(split+".csv",[r for r in inventory if r["split"]==split and r["eligibility"]=="eligible_candidate_review"])
    sets={s:{r["group_id"] for r in inventory if r["split"]==s and r["eligibility"]=="eligible_candidate_review"} for s in ("train","validation","test")}
    print(f"Images found: {len(images):,}")
    print("Eligible candidate images:",sum(len([r for r in inventory if r["split"]==s and r["eligibility"]=="eligible_candidate_review"]) for s in ("train","validation","test")))
    print("Group overlap check:",not (sets["train"]&sets["validation"] or sets["train"]&sets["test"] or sets["validation"]&sets["test"]))
    print("WARNING: SOCOFing filename IDs are heuristic; FVC partition grouping is not identity-verified.")
    print("Manifests saved:",out.resolve())
if __name__=="__main__": main()
