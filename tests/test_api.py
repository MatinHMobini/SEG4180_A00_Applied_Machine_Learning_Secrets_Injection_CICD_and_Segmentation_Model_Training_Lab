import io
from PIL import Image
from app import app

def test_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.is_json

def test_predict_requires_image():
    client = app.test_client()
    r = client.post("/predict")
    assert r.status_code == 400

def test_predict_returns_png():
    client = app.test_client()
    img = Image.new("RGB", (128, 128), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/predict",
        data={"image": (buf, "test.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.content_type == "image/png"