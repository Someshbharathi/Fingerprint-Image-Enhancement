from __future__ import annotations
import csv, json, logging, threading, uuid, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from src.dataset import FingerprintEnhancementDataset
from src.preprocessing import tensor_to_image
from .metrics import calculate_metrics

log = logging.getLogger(__name__)

class EvaluationManager:
    def __init__(self, model_service, manifest: Path, results_dir: Path):
        self.model = model_service; self.manifest = Path(manifest); self.results = Path(results_dir)
        self.results.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock(); self.jobs = {}; self.active = None

    def persisted(self):
        runs=[]
        for path in sorted(self.results.glob("evaluation_*.json")):
            try:
                run=json.loads(path.read_text(encoding="utf-8"))
                if isinstance(run,dict): runs.append(run)
                else: log.warning("Ignoring invalid evaluation record %s: expected a JSON object",path.name)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Ignoring unreadable evaluation record %s: %s",path.name,exc)
        # Filenames contain random run IDs, so filename order does not mean
        # chronological order. The overview uses the final record as latest.
        runs.sort(key=lambda run: (str(run.get("created_at", "")), str(run.get("run_id", ""))))
        return runs

    def start(self):
        with self.lock:
            if self.active and self.jobs.get(self.active, {}).get("status") == "running":
                return self.active, False
            if self.model.model is None: raise RuntimeError("Model is unavailable: " + str(self.model.error))
            if not self.manifest.is_file(): raise FileNotFoundError(f"Test manifest not found: {self.manifest}")
            job = uuid.uuid4().hex
            self.jobs[job] = {"job_id": job, "status": "running", "progress": 0, "error": None}
            self.active = job
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return job, True

    def _run(self, job):
        state = self.jobs[job]
        try:
            with self.manifest.open("r", newline="", encoding="utf-8-sig") as f:
                paths = [r["path"] for r in csv.DictReader(f) if r.get("split") == "test" and r.get("eligibility") in ("eligible_candidate_review", "eligible_candidate")]
            if not paths: raise ValueError("The held-out test manifest has no eligible test rows")
            missing=[path for path in paths if not Path(path).is_file()]
            available=[path for path in paths if Path(path).is_file()]
            if not available: raise FileNotFoundError(f"None of the {len(paths)} eligible test image paths exist; verify paths in {self.manifest}")
            size = self.model.image_size
            ds = FingerprintEnhancementDataset(paths=available, image_size=size, training=False, evaluation_degradation=True, seed=1234)
            rows=[]; failures=[{"path":path,"error":"Test image path does not exist"} for path in missing]; latency=[]
            for idx in range(len(ds)):
                try:
                    sample=ds[idx]; x=sample["input"].unsqueeze(0).to(self.model.device)
                    start=time.perf_counter()
                    with self.model.lock, torch.inference_mode(): y=self.model.model(x)
                    ms=(time.perf_counter()-start)*1000; latency.append(ms)
                    enhanced=y[0,0].detach().cpu().numpy()
                    baseline=sample["input"][0].detach().cpu().numpy()
                    reference=sample["target"][0].detach().cpu().numpy()
                    baseline_metrics=calculate_metrics(baseline,reference)
                    enhanced_metrics=calculate_metrics(enhanced,reference)
                    # Keep the historical top-level metric keys as enhanced-versus-reference
                    # aliases for existing consumers; the nested groups are authoritative.
                    row={"image":Path(sample["path"]).name,"path":sample["path"],
                         "baseline":baseline_metrics,"enhanced":enhanced_metrics,
                         **enhanced_metrics,"inference_ms":ms}
                    rows.append(row)
                except Exception as exc: failures.append({"path":paths[idx],"error":str(exc)})
                state["progress"] = int((len(missing)+idx+1)/len(paths)*100)
            if not rows: raise RuntimeError("No test samples could be evaluated")
            def mean_metric(values):
                valid=[v for v in values if v is not None and np.isfinite(v)]
                return float(np.mean(valid)) if valid else None
            baseline_summary={k:mean_metric([r["baseline"][k] for r in rows]) for k in ("mse","psnr_db","ssim")}
            enhanced_summary={k:mean_metric([r["enhanced"][k] for r in rows]) for k in ("mse","psnr_db","ssim")}
            latency_mean=mean_metric([r["inference_ms"] for r in rows])
            summary={"baseline":baseline_summary,"enhanced":enhanced_summary,"inference_ms":latency_mean,
                     # Legacy aggregate aliases retain their original enhanced meaning.
                     **enhanced_summary}
            run={"run_id":job,"created_at":datetime.now(timezone.utc).isoformat(),"status":"completed","checkpoint":self.model.path.name,
                 "protocol":"Existing test.csv paths; grayscale resize/pad; deterministic evaluation_degradation seed=1234; baseline is degraded input and enhanced is model output, both compared with the clean target.",
                 "metric_scope":"baseline and enhanced metrics are independently compared with the clean reference.",
                 "intensity_range":"[0,1]","aggregation":"Arithmetic mean of baseline and enhanced per-image metrics over the same successfully evaluated images; latency mean per image.",
                 "sample_count":len(rows),"failed_count":len(failures),"summary":summary,"rows":rows,"failures":failures}
            target=self.results/f"evaluation_{job}.json"
            temporary=target.with_suffix(".tmp")
            temporary.write_text(json.dumps(run,indent=2,allow_nan=False),encoding="utf-8")
            temporary.replace(target)
            state.update({"status":"completed","progress":100,"run_id":job,"result":run})
        except Exception as exc: state.update({"status":"failed","error":str(exc)})

    def status(self, job): return self.jobs.get(job, {"job_id":job,"status":"unknown"})
