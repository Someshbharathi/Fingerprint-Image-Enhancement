import io
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from backend.app import create_app
import backend.app as app_module
from backend.config import settings as default_settings
from backend.metrics import calculate_metrics
from backend.model_service import ModelService
from backend.evaluation import EvaluationManager
from src.dataset import FingerprintEnhancementDataset
from src.preprocessing import image_to_tensor

class DummyModel:
    model=object(); error=None; image_size=16; device="cpu"; path=__import__('pathlib').Path("dummy.pt")
    info={"architecture":"test","checkpoint_exists":True,"loaded":True,"expected_size":[16,16],"channels":1,"device":"cpu"}
    latest_inference=None
    def status(self): return {**self.info,"ready":True,"error":None}
    decode=staticmethod(ModelService.decode)
    def enhance(self,image): return Image.new("L",(16,16),128),0.012,list(image.size)

class EmptyEval:
    def persisted(self): return []
    def start(self): raise FileNotFoundError("test split missing")
    def status(self,job): return {"job_id":job,"status":"unknown"}

@pytest.fixture
def client():
    return create_app(DummyModel(),EmptyEval()).test_client()

def image_bytes(fmt="PNG"):
    out=io.BytesIO(); Image.new("L",(8,6),128).save(out,format=fmt); return out.getvalue()

def tiff_bytes(multi_frame=False,size=(11,7)):
    out=io.BytesIO()
    first=Image.new("L",size,64)
    if multi_frame: first.save(out,format="TIFF",save_all=True,append_images=[Image.new("L",size,200)])
    else: first.save(out,format="TIFF")
    return out.getvalue()

def test_health_endpoint(client):
    response=client.get("/api/health")
    assert response.status_code==200 and response.json["ready"] is True

def test_upload_validation(client):
    assert client.post("/api/enhance").status_code==400
    invalid=client.post("/api/enhance",data={"image":(io.BytesIO(b"bad"),"x.png")},content_type="multipart/form-data")
    assert invalid.status_code==400
    unsupported=client.post("/api/enhance",data={"image":(io.BytesIO(b"bad"),"x.gif")},content_type="multipart/form-data")
    assert unsupported.status_code==400

def test_inference_error_is_safe_for_client_and_logged(caplog):
    class BrokenModel(DummyModel):
        def enhance(self,image): raise RuntimeError("private model stack detail")
    response=create_app(BrokenModel(),EmptyEval()).test_client().post(
        "/api/enhance",data={"image":(io.BytesIO(image_bytes()),"finger.png")},content_type="multipart/form-data")
    assert response.status_code==500
    assert "private model stack detail" not in response.get_data(as_text=True)
    assert "private model stack detail" in caplog.text

def test_enhance_returns_image(client):
    response=client.post("/api/enhance",data={"image":(io.BytesIO(image_bytes()),"finger.png")},content_type="multipart/form-data")
    assert response.status_code==200 and response.mimetype=="image/png"
    assert response.headers["X-Output-Size"]=="16x16"

@pytest.mark.parametrize(("filename","contents"),[("finger.tif",tiff_bytes()),("finger.tiff",tiff_bytes()),("finger.png",image_bytes("PNG")),("finger.jpg",image_bytes("JPEG")),("finger.jpeg",image_bytes("JPEG"))])
def test_supported_image_uploads_enhance(client,filename,contents):
    response=client.post("/api/enhance",data={"image":(io.BytesIO(contents),filename)},content_type="multipart/form-data")
    assert response.status_code==200 and response.mimetype=="image/png"

def test_tiff_multiframe_uses_first_frame():
    decoded=ModelService.decode(tiff_bytes(multi_frame=True),"finger.tiff")
    assert decoded.size==(11,7) and decoded.getpixel((0,0))==64

def test_decode_accepts_bytes_stream_and_pillow_image():
    raw=tiff_bytes()
    from_bytes=ModelService.decode(raw,"finger.tif")
    from_stream=ModelService.decode(io.BytesIO(raw),"finger.tif")
    with Image.open(io.BytesIO(raw)) as opened:
        already_decoded=ModelService.decode(opened)
    assert from_bytes.mode==from_stream.mode==already_decoded.mode=="L"
    assert np.array_equal(np.asarray(from_bytes),np.asarray(from_stream))
    assert np.array_equal(np.asarray(from_bytes),np.asarray(already_decoded))

def test_inference_receives_expected_tensor_from_decoded_tiff():
    class CaptureModel:
        def __call__(self,tensor):
            self.input=tensor.detach().clone()
            return tensor
    service=ModelService.__new__(ModelService)
    service.model=CaptureModel(); service.error=None; service.image_size=16
    service.device=torch.device("cpu"); service.lock=__import__('threading').Lock()
    service.latest_inference=None
    image=ModelService.decode(tiff_bytes(),"finger.tif")
    output,_,_=service.enhance(image)
    tensor=service.model.input
    assert tuple(tensor.shape)==(1,1,16,16)
    assert tensor.dtype==torch.float32 and 0<=float(tensor.min())<=float(tensor.max())<=1
    assert output.mode=="L" and output.size==(16,16)

def test_tiff_endpoint_reaches_real_checkpoint_and_returns_image(tmp_path,monkeypatch):
    if not default_settings.checkpoint.is_file():
        pytest.skip("Project checkpoint is not configured in this environment")
    service=ModelService(default_settings.checkpoint)
    if service.model is None:
        pytest.skip(f"Project checkpoint could not load: {service.error}")
    source=Path(__file__).resolve().parents[1]/"datasets"/"FVC2000"/"Dbs"/"Db1_b"/"105_1.tif"
    if not source.is_file():
        pytest.skip("Repository TIFF sample 105_1.tif is not present")
    with Image.open(source) as sample:
        assert sample.size==(300,300)
        sample.load()
    monkeypatch.setattr(app_module,"settings",SimpleNamespace(max_upload_mb=12,results_dir=tmp_path))
    client=create_app(service,EmptyEval()).test_client()
    response=client.post("/api/enhance",data={"image":(io.BytesIO(source.read_bytes()),source.name)},content_type="multipart/form-data")
    assert response.status_code==200, response.get_data(as_text=True)
    assert service.latest_inference is not None
    with Image.open(io.BytesIO(response.data)) as output:
        output.load()
        assert output.format=="PNG" and output.mode=="L" and output.size==(service.image_size,service.image_size)

def test_metrics_known_arrays():
    a=np.zeros((32,32)); b=np.ones((32,32))
    result=calculate_metrics(a,b)
    assert result["mse"]==1 and result["psnr_db"]==0
    identical=calculate_metrics(a,a)
    assert identical["mse"]==0 and identical["psnr_db"] is None and identical["ssim"]==1
    with pytest.raises(ValueError): calculate_metrics(np.zeros((8,8)),np.zeros((7,8)))

def test_preprocessing_shape_and_range():
    tensor=image_to_tensor(Image.new("L",(13,7),128),16)
    assert tuple(tensor.shape)==(1,16,16) and 0<=float(tensor.min())<=float(tensor.max())<=1

def test_dataset_pairing_and_determinism(tmp_path):
    p=tmp_path/"clean.png"; Image.new("L",(32,32),180).save(p)
    ds=FingerprintEnhancementDataset(paths=[p],image_size=32,training=False,evaluation_degradation=True,seed=1234)
    first=ds[0]; second=ds[0]
    assert first["path"]==str(p) and np.array_equal(first["target"].numpy(),second["target"].numpy())
    assert np.array_equal(first["input"].numpy(),second["input"].numpy())

def test_evaluation_endpoint_missing_dataset(client):
    response=client.post("/api/evaluate")
    assert response.status_code==404 and response.json["code"]=="dataset_unavailable"

def test_empty_analytics(client):
    response=client.get("/api/analytics")
    assert response.status_code==200 and response.json["evaluations"]==[]

def test_missing_checkpoint_is_unavailable(tmp_path):
    service=ModelService(tmp_path/"absent.pt")
    assert service.model is None and service.status()["checkpoint_exists"] is False

def test_persisted_evaluation_retrieval(tmp_path):
    manager=EvaluationManager(DummyModel(),tmp_path/"test.csv",tmp_path/"runs")
    expected={"run_id":"real-run","status":"completed","rows":[]}
    (tmp_path/"runs"/"evaluation_real-run.json").write_text(__import__('json').dumps(expected),encoding="utf-8")
    assert manager.persisted()==[expected]

def test_persisted_evaluation_skips_damaged_record(tmp_path,caplog):
    manager=EvaluationManager(DummyModel(),tmp_path/"test.csv",tmp_path/"runs")
    (tmp_path/"runs"/"evaluation_valid.json").write_text(json.dumps({"run_id":"valid"}),encoding="utf-8")
    (tmp_path/"runs"/"evaluation_partial.json").write_text('{"run_id":',encoding="utf-8")
    assert manager.persisted()==[{"run_id":"valid"}]
    assert "Ignoring unreadable evaluation record" in caplog.text

def test_persisted_evaluations_are_ordered_by_creation_time(tmp_path):
    manager=EvaluationManager(DummyModel(),tmp_path/"test.csv",tmp_path/"runs")
    for filename,run_id,created_at in (
        ("evaluation_a.json","random-id-z","2026-10-22T10:00:00+00:00"),
        ("evaluation_z.json","random-id-a","2026-10-23T10:00:00+00:00"),
    ):
        (tmp_path/"runs"/filename).write_text(json.dumps({"run_id":run_id,"created_at":created_at}),encoding="utf-8")
    assert [run["run_id"] for run in manager.persisted()]==["random-id-z","random-id-a"]

def test_evaluation_counts_missing_manifest_paths_as_failures(tmp_path):
    image_path=tmp_path/"clean.png"; Image.new("L",(16,16),180).save(image_path)
    manifest=tmp_path/"test.csv"
    with manifest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["path","split","eligibility"]); writer.writeheader()
        writer.writerow({"path":str(image_path),"split":"test","eligibility":"eligible_candidate_review"})
        writer.writerow({"path":str(tmp_path/"missing.png"),"split":"test","eligibility":"eligible_candidate_review"})

    class IdentityModel:
        def __call__(self,x): return x
    class EvaluationService:
        model=IdentityModel(); device=torch.device("cpu"); image_size=16
        lock=__import__("threading").Lock(); path=tmp_path/"checkpoint.pt"; error=None

    manager=EvaluationManager(EvaluationService(),manifest,tmp_path/"runs")
    manager.jobs["audit-run"]={"job_id":"audit-run","status":"running","progress":0}
    manager._run("audit-run")
    status=manager.status("audit-run")
    assert status["status"]=="completed"
    assert status["result"]["sample_count"]==1
    assert status["result"]["failed_count"]==1
    assert status["result"]["failures"][0]["path"].endswith("missing.png")
    assert len(manager.persisted())==1
