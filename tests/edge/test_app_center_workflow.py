import json
import os
import threading
import time
from types import SimpleNamespace

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference.edge.api import create_app
from inference.edge.settings import EdgeSettings
from inference.edge.video import video_error


class Manager:
    def clear(self):
        pass

    def list_models(self):
        return []


SPEC = {"version": "1.0", "inputs": [{"type": "WorkflowImage", "name": "scene"}],
        "steps": [], "outputs": [{"type": "JsonField", "name": "view", "selector": "$inputs.scene"}]}


def test_app_center_rtsp_deployment_reads_selected_source_and_keeps_results(tmp_path):
    opened = []
    closed = threading.Event()

    class Source:
        def frames(self):
            while not closed.wait(0.01):
                yield SimpleNamespace(data=np.zeros((12, 16, 3), np.uint8), fmt="BGR", pts=time.monotonic())

        def cancel(self):
            closed.set()

        close = cancel

    def source(*args):
        opened.append(args)
        return Source()

    secret_url = "rtsp://test-user:test-password@camera.invalid/live"
    settings = EdgeSettings(storage_root=tmp_path, workflow_autostart=True,
                            workflow_id="scene", video_source="rtsp", rtsp_url=secret_url,
                            workflow_fps=2, api_token="device-test")
    app = create_app(settings, manager=Manager(), source_factory=source)
    app.state.workflow_store.save("scene", {"specification": SPEC})
    with TestClient(app) as client:
        assert client.get("/app-center/workflow-runtime").status_code == 401
        headers = {"X-Inference-Token": "device-test"}
        deadline = time.monotonic() + 5
        while not app.state.video.status()["frames"] and time.monotonic() < deadline:
            time.sleep(.01)
        result = client.get("/app-center/workflow-runtime?include_result=true", headers=headers)
        assert result.status_code == 200
        data = result.json()
        assert opened == [(secret_url, {})]
        assert data["deployment"]["video_source"] == "rtsp"
        assert data["deployment"]["target_fps"] == 2
        assert data["latest"]["result"][0]["view"]["type"] == "base64"
        assert "test-password" not in result.text
        assert app.state.video.latest_result() is not None
        # Status polling has no image payload, and never drains the SDK buffer.
        assert client.get("/app-center/workflow-runtime", headers=headers).json()["latest"] is None
        assert app.state.video.results()
    assert closed.is_set()


@pytest.mark.parametrize("url", ["", "http://camera/live", "rtsp:///stream", "rtsp://host:invalid/live"])
def test_invalid_rtsp_fails_before_opening_video(tmp_path, url):
    def forbidden(*args):
        raise AssertionError("Invalid URLs must never open a camera")

    app = create_app(EdgeSettings(storage_root=tmp_path, workflow_autostart=True, workflow_id="scene",
                                 video_source="rtsp", rtsp_url=url), manager=Manager(), source_factory=forbidden)
    app.state.workflow_store.save("scene", {"specification": SPEC})
    with TestClient(app) as client:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = client.get("/workflow-deployment").json()
            if status["status"] == "failed":
                break
            time.sleep(.01)
        assert status["error"]["code"] == "invalid_rtsp_url"


def test_media_errors_never_publish_rtsp_credentials():
    error = RuntimeError("Decode failed for 'rtsp://alice:private@camera/live?token=secret'")
    assert video_error(error) == "Decode failed for '[RTSP source]'"
