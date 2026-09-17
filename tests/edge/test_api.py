import base64
import os
import time
from types import SimpleNamespace
from threading import Event

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference.edge.api import create_app
from inference.edge.errors import EdgeError
from inference.edge.settings import EdgeSettings
from inference.edge.video import VideoRunner


class FakeManager:
    def __init__(self):
        self.cleared = False
        self.images = []

    def infer(self, model_id, image, **kwargs):
        self.images.append((image.shape, image[0, 0].tolist()))
        return {"predictions": [], "image": {"width": image.shape[1], "height": image.shape[0]}}

    def add_model(self, model_id):
        pass

    def remove(self, model_id):
        raise AssertionError("Active camera must prevent model removal")

    def clear(self):
        self.cleared = True

    def list_models(self):
        return []


def encoded_image(shape=(24, 32, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :, 0] = 255
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return {"type": "base64", "value": base64.b64encode(encoded).decode()}


def test_http_inference_is_bgr_and_lifespan_releases_models():
    manager = FakeManager()
    with TestClient(create_app(manager=manager)) as client:
        assert client.get("/healthz").json()["backend"] == "rknn"
        response = client.post("/infer/object_detection", json={"model_id": "detector/1", "image": encoded_image()})
        assert response.status_code == 200, response.text
        assert response.json()["image"] == {"width": 32, "height": 24}
        assert manager.images == [((24, 32, 3), [255, 0, 0])]
        assert client.post("/model-conversion", json={"onnx": "any"}).status_code == 503
    assert manager.cleared


def test_resource_limits_reject_before_inference():
    manager = FakeManager()
    settings = EdgeSettings(max_request_bytes=1024, max_image_pixels=256)
    with TestClient(create_app(settings, manager=manager)) as client:
        assert client.post("/infer/object_detection", content=b"x" * 1025).status_code == 413
        assert client.post("/infer/object_detection", json={"model_id": "detector/1", "image": encoded_image()}).status_code == 413
        assert client.post("/infer/object_detection", json={"model_id": "detector/1", "image": {"type": "url", "value": "file:///etc/passwd"}}).status_code == 422
        assert client.post("/infer/object_detection", json={"model_id": "detector/1", "image": encoded_image((4, 4, 3)), "confidence": 2}).status_code == 422
    assert manager.images == []


def test_exposed_listener_requires_explicit_token_and_authentication():
    with pytest.raises(ValueError, match="token|TOKEN"):
        EdgeSettings(host="0.0.0.0")
    settings = EdgeSettings(host="0.0.0.0", api_token="test-only")
    with TestClient(create_app(settings, manager=FakeManager())) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/capabilities").status_code == 401
        response = client.get("/capabilities", headers={"Authorization": "Bearer test-only"})
        assert response.status_code == 200
        assert response.json()["training"] is False


def test_camera_keeps_two_results_without_retaining_frames():
    class Source:
        closed = False

        def frames(self):
            for index in range(6):
                time.sleep(0.025)
                yield SimpleNamespace(data=np.zeros((8, 8, 3), np.uint8), pts=index / 10, pts_us=index * 100000)

        def close(self):
            self.closed = True

    class Workflows:
        def clear(self, **kwargs):
            pass

    source = Source()
    video = VideoRunner(FakeManager(), Workflows(), EdgeSettings(max_fps=60), source_factory=lambda: source)
    video.start(model_id="detector/1")
    with pytest.raises(EdgeError, match="one"):
        video.start(model_id="detector/1")
    video._thread.join(timeout=2)
    assert source.closed
    assert video.status()["frames"] == 6
    results = video.results()
    assert [r["frame_id"] for r in results] == [4, 5]
    assert "data" not in results[0]
    video.stop()
    assert video.status()["retained_results"] == 0


def test_active_camera_blocks_http_model_changes_even_while_paused():
    closed = Event()

    class Source:
        def frames(self):
            while not closed.wait(0.02):
                yield SimpleNamespace(data=np.zeros((8, 8, 3), np.uint8), pts_us=0)

        def close(self):
            closed.set()

    manager = FakeManager()
    with TestClient(create_app(manager=manager, source_factory=Source)) as client:
        response = client.post("/inference_pipelines/initialise", json={"model_id": "detector/1"})
        assert response.status_code == 200, response.text
        pipeline = {"pipeline_id": response.json()["pipeline_id"]}
        assert client.post("/inference_pipelines/pause", json=pipeline).status_code == 200
        for path, body in (
            ("/model/add", {"model_id": "another/1"}),
            ("/model/remove", {"model_id": "detector/1"}),
            ("/model/clear", {}),
            ("/infer/object_detection", {"model_id": "another/1", "image": encoded_image()}),
            ("/workflows/run", {"specification": {}, "inputs": {}}),
        ):
            response = client.post(path, json=body)
            assert response.status_code == 409, (path, response.text)
        assert client.post("/inference_pipelines/terminate", json=pipeline).status_code == 200
        assert closed.is_set()
        assert client.post("/model/clear").status_code == 200
