import base64
import os
import threading
import time
from types import SimpleNamespace

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference.edge.api import create_app
from inference.edge.errors import EdgeError
from inference.edge.preview import WorkflowPreview
from inference.edge.settings import EdgeSettings

PIPELINE = "a" * 32


def record(image=None, index=1, pipeline=PIPELINE):
    return {
        "pipeline_id": pipeline,
        "frame_id": index,
        "frame_timestamp": "2026-09-17T12:00:00Z",
        "result": [
            {"detections": {"image": {"width": 8, "height": 8}}, "visual": image}
        ],
    }


def encoded(color=100):
    ok, data = cv2.imencode(".jpg", np.full((8, 8, 3), color, np.uint8))
    assert ok
    return {"type": "base64", "value": base64.b64encode(data).decode()}


def test_shared_output_and_conditional_absence_never_return_old_image():
    preview = WorkflowPreview(1024 * 1024)
    preview.reset(PIPELINE)
    preview.publish(record(encoded()), None)
    outputs = preview.status()["outputs"]
    assert len(outputs) == 1 and outputs[0]["name"] == "visual"
    key = outputs[0]["id"]
    code, image, content_type, headers = preview.read(PIPELINE, key)
    assert code == 200 and content_type == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (
        8,
        8,
        3,
    )
    assert (
        preview.read(PIPELINE, key)[1] is image
    )  # Shared bytes, not per-viewer copies.
    assert preview.read(PIPELINE, key, headers["ETag"])[0] == 304
    preview.publish(record(None, index=2), None)
    assert preview.status()["outputs"][0]["available"] is False
    with pytest.raises(EdgeError) as error:
        preview.read(PIPELINE, key, headers["ETag"])
    assert error.value.status_code == 425
    preview.reset("b" * 32)
    assert preview.status()["outputs"] == []
    with pytest.raises(EdgeError) as error:
        preview.read(PIPELINE, key)
    assert error.value.status_code == 409


def test_original_is_captured_before_mutation_and_expires(monkeypatch):
    preview = WorkflowPreview(1024 * 1024)
    preview.reset(PIPELINE)
    image = np.zeros((8, 8, 3), np.uint8)
    assert preview.capture_original(image) is None
    with pytest.raises(EdgeError):
        preview.read(PIPELINE, "original")
    original = preview.capture_original(image)
    image[:] = 255
    preview.publish(record(encoded(255)), original)
    data = preview.read(PIPELINE, "original")[1]
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).max() == 0
    now = time.monotonic()
    monkeypatch.setattr("inference.edge.preview.time.monotonic", lambda: now + 20)
    assert preview.capture_original(image) is None
    preview.status()
    assert preview._original is None


def test_output_bounds_and_raster_only():
    preview = WorkflowPreview(100)
    preview.reset(PIPELINE)
    value = record({"type": "base64", "value": base64.b64encode(b"<svg/>").decode()})
    preview.publish(value, None)
    key = preview.status()["outputs"][0]["id"]
    with pytest.raises(EdgeError) as error:
        preview.read(PIPELINE, key)
    assert error.value.status_code == 422
    preview.publish(record(encoded()), None)
    with pytest.raises(EdgeError) as error:
        preview.read(PIPELINE, key)
    assert error.value.status_code == 413
    value["result"] = [{str(i): encoded() for i in range(40)}]
    preview.publish(value, None)
    assert len(preview.status()["outputs"]) <= 16


def test_api_shares_running_graph_preserves_raw_input_and_requires_device_auth():
    advance, closed = threading.Event(), threading.Event()

    class Source:
        def frames(self):
            while not closed.is_set():
                if not advance.wait(0.05):
                    continue
                advance.clear()
                if not closed.is_set():
                    yield SimpleNamespace(data=np.zeros((8, 8, 3), np.uint8), pts_us=1)

        def close(self):
            closed.set()

        cancel = close

    class Manager:
        def clear(self):
            pass

    class Workflows:
        calls = 0

        def validate(self, *args, **kwargs):
            pass

        def preflight_models(self, *args, **kwargs):
            pass

        def clear(self, **kwargs):
            pass

        def run(self, specification, inputs, **kwargs):
            self.calls += 1
            inputs["image"][:] = 255  # Equivalent to copy_image=False.
            return [{"annotated": encoded(255)}]

    app = create_app(
        EdgeSettings(host="0.0.0.0", api_token="test-preview"),
        manager=Manager(),
        source_factory=Source,
    )
    workflows = Workflows()
    app.state.video.workflows = workflows
    headers = {"X-Inference-Token": "test-preview"}
    with TestClient(app) as client:
        pipeline = app.state.video.start(
            specification={"inputs": [{"type": "WorkflowImage", "name": "image"}]}
        )["pipeline_id"]
        app.state.deployment._pipeline_id = pipeline
        advance.set()
        deadline = time.monotonic() + 2
        while app.state.video.status()["frames"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        metadata = client.get("/app-center/workflow-runtime", headers=headers).json()
        assert metadata["latest"] is None
        output = metadata["preview"]["outputs"][0]["id"]
        url = f"/app-center/workflow-preview?pipeline_id={pipeline}&output={output}"
        assert client.get(url).status_code == 401
        first = client.get(url, headers=headers)
        assert first.status_code == 200
        assert (
            client.get(
                url, headers={**headers, "If-None-Match": first.headers["etag"]}
            ).status_code
            == 304
        )
        assert client.get(url, headers=headers).content == first.content
        assert workflows.calls == 1
        raw_url = f"/app-center/workflow-preview?pipeline_id={pipeline}&output=original"
        assert client.get(raw_url, headers=headers).status_code == 425
        time.sleep(0.35)
        advance.set()
        deadline = time.monotonic() + 2
        while app.state.video.status()["frames"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raw = client.get(raw_url, headers=headers)
        assert raw.status_code == 200
        assert (
            cv2.imdecode(np.frombuffer(raw.content, np.uint8), cv2.IMREAD_COLOR).max()
            == 0
        )
        assert workflows.calls == 2
        app.state.deployment._pipeline_id = "b" * 32
        assert client.get(url, headers=headers).status_code == 409
