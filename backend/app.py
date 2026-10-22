import logging, io, json
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from .config import settings
from .model_service import ModelService
from .evaluation import EvaluationManager

log = logging.getLogger(__name__)

def create_app(model_service=None, evaluation_manager=None):
    app=Flask(__name__); app.config["MAX_CONTENT_LENGTH"]=settings.max_upload_mb*1024*1024
    CORS(app, resources={r"/api/*":{"origins":["http://localhost:8501","http://127.0.0.1:8501"]}})
    service=model_service or ModelService(settings.checkpoint)
    evaluator=evaluation_manager or EvaluationManager(service,settings.test_manifest,settings.results_dir)
    app.extensions["model_service"]=service; app.extensions["evaluation_manager"]=evaluator
    @app.errorhandler(413)
    def too_large(_): return jsonify(error="Upload exceeds configured size limit",code="file_too_large"),413
    @app.get("/api/health")
    def health():
        dataset_path=getattr(evaluator,"manifest",settings.test_manifest)
        return jsonify(backend="ready",model=service.status(),ready=service.model is not None,
                       dataset={"manifest_path":str(dataset_path),"manifest_exists":dataset_path.is_file()})
    @app.post("/api/enhance")
    def enhance():
        if "image" not in request.files: return jsonify(error="Multipart field 'image' is required",code="missing_image"),400
        f=request.files["image"]
        if not f.filename: return jsonify(error="A filename is required",code="missing_filename"),400
        try:
            image=service.decode(f.read(settings.max_upload_mb*1024*1024+1),f.filename)
            if service.model is None: return jsonify(error=service.error,code="model_unavailable"),503
            result,elapsed,input_size=service.enhance(image); buf=io.BytesIO(); result.save(buf,format="PNG"); buf.seek(0)
            response=send_file(buf,mimetype="image/png",download_name="enhanced.png")
            response.headers["X-Inference-Seconds"]=str(elapsed); response.headers["X-Input-Size"]=f"{input_size[0]}x{input_size[1]}"
            response.headers["X-Output-Size"]=f"{result.width}x{result.height}"; response.headers["X-Output-Range"]=f"{int(npmin(result))},{int(npmax(result))}"
            history_path=settings.results_dir/"processing.jsonl"; history_path.parent.mkdir(parents=True,exist_ok=True)
            with history_path.open("a",encoding="utf-8") as log_file:
                log_file.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"filename":f.filename,"seconds":elapsed,"status":"success"})+"\n")
            return response
        except ValueError as exc: return jsonify(error=str(exc),code="invalid_image"),400
        except Exception:
            # Keep the complete traceback in the Flask development terminal;
            # return a stable, non-sensitive message to dashboard users.
            log.exception("Enhancement failed")
            return jsonify(error="Enhancement failed due to an internal inference error. Check the backend terminal for details.",code="inference_failed"),500
    @app.get("/api/analytics")
    def analytics():
        runs=evaluator.persisted(); history=settings.results_dir/"processing.jsonl"; events=[]
        if history.is_file():
            for line in history.read_text(encoding="utf-8").splitlines():
                try: events.append(json.loads(line))
                except ValueError: continue
        times=[e["seconds"] for e in events if e.get("status")=="success"]
        return jsonify(evaluations=runs,processing={"images_processed":len(events),"successful_runs":len(times),"mean_inference_seconds":sum(times)/len(times) if times else None,"recent":events[-10:]})
    @app.post("/api/evaluate")
    def evaluate():
        try:
            job,created=evaluator.start(); return jsonify(job_id=job,created=created,status=evaluator.status(job)),(202 if created else 200)
        except FileNotFoundError as exc: return jsonify(error=str(exc),code="dataset_unavailable"),404
        except Exception as exc: return jsonify(error=str(exc),code="evaluation_unavailable"),503
    @app.get("/api/evaluate/<job_id>")
    def evaluation_status(job_id): return jsonify(evaluator.status(job_id))
    return app

def npmin(image): return min(image.getextrema())
def npmax(image): return max(image.getextrema())

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    log.info("FPrint backend source: %s",__file__)
    log.info("FPrint model decoder source: %s",ModelService.decode.__code__.co_filename)
    create_app().run(host=settings.host,port=settings.port,threaded=True)
