import requests

def get_json(base, path, timeout=5):
    r=requests.get(base.rstrip("/")+path,timeout=timeout); r.raise_for_status(); return r.json()

def post_image(base, data, filename, timeout=180):
    r=requests.post(base.rstrip("/")+"/api/enhance",files={"image":(filename,data)},timeout=timeout)
    if not r.ok:
        try: message=r.json().get("error",r.text)
        except ValueError: message=r.text
        raise RuntimeError(message)
    return r.content, {"seconds":float(r.headers["X-Inference-Seconds"]),"input_size":r.headers.get("X-Input-Size"),"output_size":r.headers.get("X-Output-Size"),"range":r.headers.get("X-Output-Range")}
