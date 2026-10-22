import pytest
from frontend import api
from frontend.state import clear_upload, select_upload

class Response:
    def __init__(self,status=200,payload=None,content=b"png",headers=None):
        self.status_code=status; self._payload=payload or {}; self.content=content
        self.headers=headers or {"X-Inference-Seconds":"0.02","X-Input-Size":"8x6","X-Output-Size":"16x16","X-Output-Range":"1,254"}
        self.ok=status<400; self.text="request failed"
    def raise_for_status(self):
        if not self.ok: raise RuntimeError(self.text)
    def json(self): return self._payload

def test_frontend_enhance_metadata(monkeypatch):
    monkeypatch.setattr(api.requests,"post",lambda *a,**k:Response())
    data,meta=api.post_image("http://localhost:5000",b"image","finger.png")
    assert data==b"png" and meta["seconds"]==0.02 and meta["output_size"]=="16x16"

def test_frontend_backend_error(monkeypatch):
    monkeypatch.setattr(api.requests,"post",lambda *a,**k:Response(503,{"error":"model unavailable"}))
    with pytest.raises(RuntimeError,match="model unavailable"):
        api.post_image("http://localhost:5000",b"image","finger.png")

def test_tiff_api_client_sends_original_upload_bytes(monkeypatch):
    captured={}
    def fake_post(url,**kwargs):
        captured["filename"],captured["data"]=kwargs["files"]["image"][:2]
        return Response()
    monkeypatch.setattr(api.requests,"post",fake_post)
    original=b"tiff-upload-bytes"
    api.post_image("http://localhost:5000",original,"105_1.tif")
    assert captured=={"filename":"105_1.tif","data":original}

def test_changing_upload_discards_previous_result():
    state={"upload":{"data":b"first","filename":"first.png","size":(8,6)},"result":{"data":b"old-output"}}
    select_upload(state,b"second","second.tif",(300,300))
    assert state["upload"]=={"data":b"second","filename":"second.tif","size":(300,300)}
    assert state["result"] is None

def test_invalid_upload_clears_input_and_result():
    state={"upload":{"data":b"first"},"result":{"data":b"old-output"}}
    clear_upload(state)
    assert state=={"upload":None,"result":None}
