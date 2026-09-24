"""Resource and lifecycle checks that do not require physical NPU ownership."""

import base64
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import numpy as np
import pytest

from inference.edge.errors import EdgeError
from inference.edge.limits import encode_bounded_json
from inference.edge.settings import EdgeSettings
from inference.edge.video import CancellableCameraSource, VideoRunner
from inference.edge.workflows import EdgeWorkflows


class BoxesManager:
    def __init__(self):
        self.calls = 0

    def infer_from_request_sync(self, model_id, request):
        self.calls += 1
        height, width = request["image"]["value"].shape[:2]
        return {
            "image": {"width": width, "height": height},
            "predictions": [
                {
                    "x": width / 2,
                    "y": height / 2,
                    "width": width,
                    "height": height,
                    "confidence": 0.9,
                    "class": "object",
                    "class_id": 0,
                }
                for _ in range(3)
            ],
        }


def detection_crop_workflow():
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "detect",
                "images": "$inputs.image",
                "model_id": "detector/1",
            },
            {
                "type": "DynamicCrop",
                "name": "crops",
                "images": "$inputs.image",
                "predictions": "$steps.detect.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "crops", "selector": "$steps.crops.crops"}
        ],
    }


def test_crop_pixel_budget_rejects_before_fanout_and_resets_next_request():
    manager = BoxesManager()
    runner = EdgeWorkflows(manager, EdgeSettings(max_image_pixels=32 * 32))
    for _ in range(2):
        with pytest.raises(EdgeError) as error:
            runner.run(
                detection_crop_workflow(), {"image": np.zeros((32, 32, 3), np.uint8)}
            )
        assert error.value.code == "resource_budget"
        assert error.value.status_code == 422
    assert manager.calls == 2


def test_video_image_output_encodes_within_cumulative_budget(monkeypatch):
    from inference.core.workflows.execution_engine.entities import base

    workflow = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [],
        "outputs": [
            {"type": "JsonField", "name": "image", "selector": "$inputs.image"}
        ],
    }
    runner = EdgeWorkflows(object(), EdgeSettings())
    result = runner.run(workflow, {"image": np.zeros((8, 8, 3), np.uint8)}, stream_id="camera")
    assert result[0]["image"]["type"] == "base64"
    assert base64.b64decode(result[0]["image"]["value"]).startswith(b"\xff\xd8")
    workflow["outputs"].append({"type": "JsonField", "name": "second", "selector": "$inputs.image"})
    runner = EdgeWorkflows(object(), EdgeSettings(max_image_pixels=64))
    with pytest.raises(EdgeError) as error:
        runner.run(workflow, {"image": np.zeros((8, 8, 3), np.uint8)}, stream_id="camera")
    assert error.value.code == "resource_budget"


def test_query_multiply_cannot_allocate_an_unbounded_sequence():
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "items"}],
        "steps": [
            {
                "type": "Expression",
                "name": "repeat",
                "data": {"items": "$inputs.items"},
                "switch": {
                    "type": "CasesDefinition", "cases": [],
                    "default": {
                        "type": "DynamicCaseResult", "parameter_name": "items",
                        "operations": [{"type": "Multiply", "other": 1000000000}],
                    },
                },
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "items", "selector": "$steps.repeat.output"}
        ],
    }
    with pytest.raises(EdgeError) as error:
        EdgeWorkflows(object(), EdgeSettings()).run(workflow, {"items": [1, 2]})
    assert error.value.status_code == 422
    assert error.value.code == "resource_budget"


def test_response_encoding_has_a_byte_budget():
    assert json.loads(encode_bounded_json({"x": "正常"}, 64)) == {"x": "正常"}
    with pytest.raises(EdgeError) as error:
        encode_bounded_json({"x": "x" * 128}, 64)
    assert error.value.code == "resource_budget"


def test_camera_with_no_frames_stops_without_cross_thread_native_close(monkeypatch):
    acquired = threading.Event()
    owner_threads, close_threads = [], []

    class Timeout(Exception):
        pass

    class NativeSource:
        def __init__(self, **kwargs):
            owner_threads.append(threading.get_ident())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            close_threads.append(threading.get_ident())

        def acquire(self):
            acquired.set()
            time.sleep(0.01)
            raise Timeout()

    class Converter:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

        def _convert(self, frame):
            raise AssertionError("No frame should be converted")

    monkeypatch.setitem(
        sys.modules,
        "recamera_ext",
        types.SimpleNamespace(FrameSource=NativeSource, AcquireTimeoutError=Timeout),
    )
    monkeypatch.setitem(
        sys.modules,
        "kit.adapters.official",
        types.SimpleNamespace(OfficialFrameSource=Converter),
    )
    monkeypatch.setitem(
        sys.modules,
        "kit.adapters.frame_source",
        types.SimpleNamespace(Frame=types.SimpleNamespace),
    )
    source = CancellableCameraSource()
    manager = types.SimpleNamespace(add_model=lambda model_id: None, clear=lambda: None)
    workflows = types.SimpleNamespace(clear=lambda **kwargs: None)
    runner = VideoRunner(
        manager, workflows, EdgeSettings(), source_factory=lambda: source
    )
    runner.start(model_id="detector/1")
    assert acquired.wait(1)
    start = time.monotonic()
    result = runner.stop()
    assert time.monotonic() - start < 1
    assert result["status"] == "stopped"
    assert owner_threads == close_threads
    assert close_threads[0] != threading.get_ident()


def test_video_eof_releases_model_and_stop_retries_a_failed_release():
    class Manager:
        def __init__(self):
            self.clear_calls = 0

        def add_model(self, model_id):
            pass

        def clear(self):
            self.clear_calls += 1
            if self.clear_calls < 3:
                raise RuntimeError("Broker temporarily unavailable")

    closed = []
    source = types.SimpleNamespace(
        frames=lambda: iter(()), close=lambda: closed.append(True)
    )
    manager = Manager()
    runner = VideoRunner(
        manager,
        types.SimpleNamespace(clear=lambda **kwargs: None),
        EdgeSettings(),
        source_factory=lambda: source,
    )
    runner.start(model_id="detector/1")
    runner._thread.join(1)
    assert not runner.is_active()
    assert closed == [True]
    assert manager.clear_calls == 1
    with pytest.raises(EdgeError, match="previous pipeline"):
        runner.start(model_id="detector/1")
    with pytest.raises(EdgeError) as error:
        runner.stop()
    assert error.value.code == "pipeline_stop_failed"
    runner.stop()
    assert manager.clear_calls == 3
    assert runner.status()["error"] == "Broker temporarily unavailable"
    # Repeated stop must not discard a model cached by a later HTTP request.
    runner.stop()
    assert manager.clear_calls == 3




def test_nested_crops_share_one_budget_across_all_steps():
    class HalvesManager(BoxesManager):
        def infer_from_request_sync(self, model_id, request):
            output = super().infer_from_request_sync(model_id, request)
            output["predictions"] = output["predictions"][:2]
            for detection in output["predictions"]:
                detection["width"] /= 2
            return output

    graph = detection_crop_workflow()
    graph["steps"].extend(
        [
            {
                "type": "ObjectDetectionModel",
                "name": "again",
                "images": "$steps.crops.crops",
                "model_id": "detector/1",
            },
            {
                "type": "DynamicCrop",
                "name": "nested",
                "images": "$steps.crops.crops",
                "predictions": "$steps.again.predictions",
            },
            {
                "type": "ObjectDetectionModel",
                "name": "third",
                "images": "$steps.nested.crops",
                "model_id": "detector/1",
            },
            {
                "type": "DynamicCrop",
                "name": "nested_again",
                "images": "$steps.nested.crops",
                "predictions": "$steps.third.predictions",
            },
        ]
    )
    graph["outputs"] = [
        {"type": "JsonField", "name": "crops", "selector": "$steps.nested_again.crops"}
    ]
    manager = HalvesManager()
    with pytest.raises(EdgeError) as error:
        EdgeWorkflows(manager, EdgeSettings(max_image_pixels=32 * 32)).run(
            graph, {"image": np.zeros((32, 32, 3), np.uint8)}
        )
    assert error.value.code == "resource_budget"
    assert manager.calls <= 7


def test_cancelled_http_calls_keep_executor_slots_until_work_finishes():
    import asyncio

    import cv2
    import httpx

    from inference.edge.api import create_app

    entered, release = threading.Event(), threading.Event()

    class Manager:
        def infer(self, model_id, image, **kwargs):
            entered.set()
            assert release.wait(3)
            return {"predictions": []}

        def clear(self):
            pass

    _, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    body = {
        "model_id": "detector/1",
        "image": {"type": "base64", "value": base64.b64encode(encoded).decode()},
    }
    app = create_app(manager=Manager())

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/infer/object_detection", json=body)
                )
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.005)
                assert entered.is_set()
                second = asyncio.create_task(
                    client.post("/infer/object_detection", json=body)
                )
                await asyncio.sleep(0.03)
                first.cancel()
                second.cancel()
                await asyncio.gather(first, second, return_exceptions=True)
                response = await client.post("/infer/object_detection", json=body)
                assert response.status_code == 429, response.text
                release.set()

    try:
        asyncio.run(run())
    finally:
        release.set()
