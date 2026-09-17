"""Original Workflow HTTP/SDK contracts with the actual NumPy execution engine."""

import base64
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import cv2
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from inference.edge.api import create_app
from inference.edge.errors import EdgeError
from inference.edge.settings import EdgeSettings
from inference.edge.storage import WorkflowStore, builder_document
from inference.edge.workflow_definitions import WorkflowDefinitions
from inference.edge.workflows import EdgeWorkflows


def parameter_workflow():
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "default_value": 7}],
        "steps": [],
        "outputs": [
            {"type": "JsonField", "name": "answer", "selector": "$inputs.value"}
        ],
    }


def image_workflow():
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "AbsoluteStaticCrop",
                "name": "crop",
                "images": "$inputs.image",
                "x_center": 4,
                "y_center": 3,
                "width": 4,
                "height": 4,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "roi", "selector": "$steps.crop.crops"}
        ],
    }


def encoded_image():
    array = np.zeros((6, 8, 3), dtype=np.uint8)
    array[:, :, 1] = 180
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return {"type": "base64", "value": base64.b64encode(encoded).decode()}


class NoModel:
    def clear(self):
        pass

    def list_models(self):
        return []


def settings(tmp_path, **kwargs):
    return EdgeSettings(storage_root=tmp_path, **kwargs)


def test_original_request_options_and_response_pass_through_engine(
    tmp_path, monkeypatch
):
    from inference.core import env
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    monkeypatch.setattr(env, "ENABLE_WORKFLOWS_PROFILING", True)
    original_init = ExecutionEngine.init
    observed = {}

    def tracked_init(**kwargs):
        observed["init"] = kwargs
        engine = original_init(**kwargs)
        original_run = engine.run

        def tracked_run(**runtime):
            observed["run"] = runtime
            return original_run(**runtime)

        engine.run = tracked_run
        return engine

    monkeypatch.setattr(ExecutionEngine, "init", tracked_init)
    with TestClient(create_app(settings(tmp_path), manager=NoModel())) as client:
        result = client.post(
            "/workflows/run",
            json={
                "specification": parameter_workflow(),
                "inputs": {},
                "api_key": None,
                "use_cache": True,
                "enable_profiling": True,
                "debug": True,
                "is_preview": True,
                "disable_sinks": True,
                "workflow_id": "test-run",
                "excluded_fields": None,
            },
        )
        assert result.status_code == 200, result.text
        data = result.json()
        assert data["outputs"] == [{"answer": 7}]
        assert data["profiler_trace"]
        assert data["python_blocks_output_streams"] is None
        assert data["python_blocks_debug_traces"] is None
        assert observed["init"]["init_parameters"]["disable_sinks"] is True
        assert observed["init"]["workflow_id"] == "test-run"
        assert observed["run"]["_is_preview"] is True


def test_original_and_legacy_drafts_named_execution_and_interface(tmp_path):
    app = create_app(settings(tmp_path), manager=NoModel())
    original = {
        "id": "original",
        "name": "Draft",
        "config": json.dumps(
            {
                "specification": parameter_workflow(),
                "layout": {"unknown_original_ui_field": [1, 2]},
            }
        ),
    }
    legacy = {
        "specification": parameter_workflow(),
        "edge_ui": {"positions": {"input": {"x": 4}}},
    }
    with TestClient(app) as client:
        for identifier, document in (("original", original), ("legacy", legacy)):
            assert (
                client.post(f"/build/api/{identifier}", json=document).status_code
                == 201
            )
            saved = client.get(f"/build/api/{identifier}").json()["data"]["config"]
            assert json.loads(saved["config"])["specification"] == parameter_workflow()
            if identifier == "original":
                assert saved["config"] == original["config"]
            for prefix in ("/local/workflows", "/infer/workflows/local"):
                response = client.post(
                    f"{prefix}/{identifier}",
                    json={"inputs": {"value": 19}, "use_cache": False},
                )
                assert response.status_code == 200, response.text
                assert response.json()["outputs"] == [{"answer": 19}]
        interface = client.post("/local/workflows/legacy/describe_interface", json={})
        assert interface.status_code == 200, interface.text
        assert set(interface.json()) == {
            "inputs",
            "outputs",
            "typing_hints",
            "kinds_schemas",
        }
        assert "value" in interface.json()["inputs"]
        # A legacy file is still recoverable; migration is a response adapter,
        # not a startup rewrite of every saved file.
        assert "config" not in app.state.workflow_store.get("legacy")["data"]["config"]


def test_batches_use_original_engine_lineage_and_cumulative_pixels(tmp_path):
    app = create_app(settings(tmp_path, max_image_pixels=96), manager=NoModel())
    with TestClient(app) as client:
        response = client.post(
            "/workflows/run",
            json={
                "specification": image_workflow(),
                "inputs": {"image": [encoded_image(), encoded_image()]},
                "use_cache": True,
                "enable_profiling": False,
            },
        )
        assert response.status_code == 200, response.text
        outputs = response.json()["outputs"]
        assert len(outputs) == 2
        for item in outputs:
            raw = base64.b64decode(item["roi"]["value"])
            assert cv2.imdecode(
                np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
            ).shape == (4, 4, 3)
        too_many = client.post(
            "/workflows/run",
            json={
                "specification": image_workflow(),
                "inputs": {"image": [encoded_image()] * 3},
            },
        )
        assert too_many.status_code == 413


def test_original_workflow_error_preserves_block_and_context(tmp_path):
    specification = image_workflow()
    specification["steps"][0]["images"] = "$steps.missing.crops"
    with TestClient(create_app(settings(tmp_path), manager=NoModel())) as client:
        response = client.post("/workflows/validate", json=specification)
        assert response.status_code == 400, response.text
        error = response.json()
        assert error["error_type"]
        assert error["context"]
        assert error["blocks_errors"]
        assert any(item["block_id"] == "crop" for item in error["blocks_errors"])


def test_inner_workflows_resolve_local_inline_and_cloud_without_heavy_loader(tmp_path):
    requested_keys = []

    def fetch(workspace, identifier, api_key, version):
        requested_keys.append(api_key)
        return {
            "workflow": {
                "id": identifier,
                "config": json.dumps({"specification": image_workflow()}),
            }
        }

    app = create_app(
        settings(tmp_path, api_token="device-only"),
        manager=NoModel(),
        workflow_fetcher=fetch,
    )
    app.state.workflow_store.save("child", {"specification": image_workflow()})
    parent = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "inner",
                "workflow_workspace_id": "local",
                "workflow_id": "child",
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "roi", "selector": "$steps.inner.roi"}
        ],
    }
    headers = {"Authorization": "Bearer device-only"}
    with TestClient(app) as client:

        def execute(spec, width=4, api_key=None):
            response = client.post(
                "/workflows/run",
                headers=headers,
                json={
                    "specification": spec,
                    "inputs": {"image": encoded_image()},
                    "api_key": api_key,
                },
            )
            assert response.status_code == 200, response.text
            encoded = response.json()["outputs"][0]["roi"]["value"]
            assert cv2.imdecode(
                np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
            ).shape == (4, width, 3)

        execute(parent)
        interface = client.post(
            "/workflows/describe_interface",
            headers=headers,
            json={"specification": parent},
        )
        assert interface.status_code == 200, interface.text
        assert interface.json()["outputs"]["roi"] == ["image"]
        # A fresh HTTP run must re-resolve a changed child before hitting the
        # original compilation cache (whose key otherwise contains only its ID).
        changed = image_workflow()
        changed["steps"][0]["width"] = 2
        app.state.workflow_store.save("child", {"specification": changed})
        execute(parent, width=2)
        inline = json.loads(json.dumps(parent))
        inline["steps"][0].pop("workflow_workspace_id")
        inline["steps"][0].pop("workflow_id")
        inline["steps"][0]["workflow_definition"] = image_workflow()
        execute(inline)
        parent["steps"][0]["workflow_workspace_id"] = "cloud-space"
        execute(parent, api_key="cloud-only")
        assert requested_keys == ["cloud-only"]


def test_cloud_definition_cache_versions_credentials_and_offline_fallback(tmp_path):
    configuration = SimpleNamespace(
        storage_root=tmp_path,
        api_token="device-secret",
        roboflow_api_key="cloud-secret",
    )
    calls = []

    def fetch(workspace, identifier, api_key, version):
        calls.append((workspace, identifier, api_key, version))
        return {
            "workflow": {
                "id": identifier,
                "config": json.dumps({"specification": parameter_workflow()}),
            }
        }

    store = WorkflowStore(tmp_path / "workflows")
    resolver = WorkflowDefinitions(store, configuration, fetcher=fetch)
    first = resolver.resolve(
        "workspace", "flow", api_key="device-secret", workflow_version_id="version-1"
    )
    first["steps"].append({"mutation": "must not corrupt cached definition"})
    assert (
        resolver.resolve("workspace", "flow", workflow_version_id="version-1")["steps"]
        == []
    )
    assert len(calls) == 1 and calls[0][2] == "cloud-secret"
    resolver.resolve("workspace", "flow", workflow_version_id="version-2")
    resolver.resolve(
        "workspace", "flow", api_key="other-cloud-key", workflow_version_id="version-1"
    )
    resolver.resolve(
        "workspace", "flow", workflow_version_id="version-1", use_cache=False
    )
    assert len(calls) == 4
    unavailable = lambda *args: (_ for _ in ()).throw(URLError("offline"))
    fresh = WorkflowDefinitions(store, configuration, fetcher=unavailable)
    assert (
        fresh.resolve("workspace", "flow", workflow_version_id="version-1")["steps"]
        == []
    )
    with pytest.raises(EdgeError) as error:
        fresh.resolve(
            "workspace", "flow", workflow_version_id="version-1", use_cache=False
        )
    assert error.value.code == "cloud_workflow_unavailable"
    configuration.roboflow_api_key = None
    with pytest.raises(EdgeError) as error:
        fresh.resolve("workspace", "flow", api_key="device-secret")
    assert error.value.code == "missing_roboflow_api_key"
    assert all(
        "secret" not in p.read_text()
        for p in (tmp_path / "cloud-workflows").glob("*.json")
    )


def test_metadata_has_separate_budget_without_truncated_docs(tmp_path):
    app = create_app(settings(tmp_path, max_response_bytes=1024), manager=NoModel())
    with TestClient(app) as client:
        response = client.post("/workflows/blocks/describe", json={})
        assert response.status_code == 200, response.text
        assert len(response.content) > 1024
        assert client.get("/workflows/definition/schema").status_code == 200


def test_cached_image_inputs_and_independent_video_metadata(tmp_path):
    specification = image_workflow()
    specification["inputs"][0] = {
        "type": "WorkflowBatchInput",
        "name": "image",
        "kind": ["image"],
        "dimensionality": 1,
    }
    runner = EdgeWorkflows(NoModel(), settings(tmp_path))
    assert (
        len(runner.run(specification, {"image": [encoded_image(), encoded_image()]}))
        == 2
    )
    specification = image_workflow()
    specification["inputs"].append({"type": "WorkflowVideoMetadata", "name": "capture"})
    specification["outputs"].append(
        {"type": "JsonField", "name": "metadata", "selector": "$inputs.capture"}
    )
    result = runner.run(
        specification,
        {"image": encoded_image()},
        stream_id="camera-one",
        video_metadata_input_name="capture",
        frame_metadata={"frame_number": 15, "fps": 10},
    )
    assert result[0]["metadata"]["frame_number"] == 15
    assert result[0]["metadata"]["video_identifier"] == "camera-one"
    runner.clear()


def test_stream_block_caches_are_reused_and_explicitly_released(tmp_path):
    from inference.core.workflows.core_steps.cache.memory_cache import (
        WorkflowMemoryCache,
    )

    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/cache_get@v1",
                "name": "before",
                "image": "$inputs.image",
                "key": "k",
            },
            {
                "type": "PropertyDefinition",
                "name": "as_text",
                "data": "$steps.before.output",
                "operations": [{"type": "ToString"}],
            },
            {
                "type": "roboflow_core/cache_set@v1",
                "name": "store",
                "image": "$inputs.image",
                "key": "k",
                "value": "$steps.as_text.output",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "before", "selector": "$steps.before.output"},
            {"type": "JsonField", "name": "stored", "selector": "$steps.store.output"},
        ],
    }
    runner = EdgeWorkflows(NoModel(), settings(tmp_path, workflow_cache_size=1))
    first = runner.run(specification, {"image": encoded_image()}, stream_id="cache-a")
    second = runner.run(specification, {"image": encoded_image()}, stream_id="cache-a")
    assert first[0]["before"] is False
    assert second[0]["before"] == "False"
    assert "cache-a" in WorkflowMemoryCache.cache
    runner.run(specification, {"image": encoded_image()}, stream_id="cache-b")
    assert "cache-a" not in WorkflowMemoryCache.cache
    assert "cache-b" in WorkflowMemoryCache.cache
    runner.clear()
    assert "cache-b" not in WorkflowMemoryCache.cache


def test_unmodified_official_sdk_against_live_http(tmp_path):
    sdk_python = Path(
        os.getenv(
            "INFERENCE_SDK_TEST_PYTHON",
            "/tmp/inference-rv1126b-sdk-test-venv/bin/python",
        )
    )
    if not sdk_python.is_file():
        pytest.skip(
            "Set INFERENCE_SDK_TEST_PYTHON to a separate environment with SDK client dependencies"
        )
    app = create_app(settings(tmp_path, api_token="device-sdk-test"), manager=NoModel())
    app.state.workflow_store.save(
        "sdk-flow", {"config": json.dumps({"specification": image_workflow()})}
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        script = """
import base64, cv2, json, sys
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
spec = json.loads(sys.argv[2])
image = np.zeros((6, 8, 3), dtype=np.uint8)
for transport in ('legacy', 'header', 'both'):
    client = InferenceHTTPClient(api_url=sys.argv[1], api_key='device-sdk-test')
    client.configure(InferenceConfiguration(api_key_transport=transport))
    direct = client.run_workflow(specification=spec, images={'image': image})
    named = client.run_workflow(workspace_name='local', workflow_id='sdk-flow', images={'image': image})
    for result in (direct, named):
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result[0]['roi']), np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (4, 4, 3), result
print('official SDK: legacy, header, both; inline and named workflows passed')
"""
        result = subprocess.run(
            [
                str(sdk_python),
                "-c",
                script,
                f"http://127.0.0.1:{port}",
                json.dumps(image_workflow()),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "INFERENCE_RUNTIME_PROFILE": "full"},
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()


def test_persisted_autostart_parameters_use_one_saved_snapshot(tmp_path):
    observed = []

    class Camera:
        def __init__(self):
            self.closed = threading.Event()

        def frames(self):
            while not self.closed.wait(0.01):
                yield SimpleNamespace(
                    data=np.zeros((6, 8, 3), dtype=np.uint8),
                    fmt="RGB",
                    pts=time.monotonic(),
                )

        def cancel(self):
            self.closed.set()

        def close(self):
            self.closed.set()

    configuration = EdgeSettings.from_app_config(
        {
            "workflow_id": "startup",
            "workflow_autostart": True,
            "workflow_parameters": '{"size":2}',
        },
        model_root=tmp_path / "models",
        storage_root=tmp_path,
    )
    specification = image_workflow()
    specification["inputs"].append(
        {"type": "WorkflowParameter", "name": "size", "default_value": 4}
    )
    specification["steps"][0]["width"] = "$inputs.size"
    camera = Camera()
    app = create_app(
        configuration,
        manager=NoModel(),
        source_factory=lambda: camera,
        on_result=observed.append,
    )
    app.state.workflow_store.save("startup", {"specification": specification})
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while len(observed) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert observed, client.get("/workflow-deployment").json()
        before = client.get("/workflow-deployment").json()
        assert before["status"] == "running"
        assert before["pipeline_id"]
        assert before["error"] is None
        specification["steps"][0]["width"] = 6
        assert (
            client.post(
                "/build/api/startup", json={"specification": specification}
            ).status_code
            == 201
        )
        configuration.workflow_parameters["size"] = 6
        count = len(observed)
        deadline = time.monotonic() + 5
        while len(observed) <= count and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(observed) > count
        encoded = observed[-1]["result"][0]["roi"]["value"]
        assert cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
        ).shape == (4, 2, 3)
        assert (
            client.get("/workflow-deployment").json()["specification_sha256"]
            == before["specification_sha256"]
        )
    assert camera.closed.is_set()


def test_autostart_is_explicit_and_failure_is_visible(tmp_path):
    def no_camera():
        raise AssertionError("A missing or disabled Workflow cannot open the camera")

    with TestClient(
        create_app(settings(tmp_path), manager=NoModel(), source_factory=no_camera)
    ) as client:
        assert client.get("/workflow-deployment").json()["status"] == "disabled"
    with TestClient(
        create_app(
            settings(tmp_path, workflow_autostart=True, workflow_id="missing"),
            manager=NoModel(),
            source_factory=no_camera,
        )
    ) as client:
        deadline = time.monotonic() + 5
        while True:
            result = client.get("/workflow-deployment").json()
            if result["status"] == "failed" or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "workflow_not_found"
        assert result["pipeline_id"] is None


def test_slow_autostart_does_not_delay_http_and_shutdown_cancels_before_camera(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    resolved = threading.Event()
    opened = []
    app = create_app(
        settings(tmp_path, workflow_autostart=True, workflow_id="slow"),
        manager=NoModel(),
        source_factory=lambda: opened.append(True),
    )

    def resolve(*args, **kwargs):
        entered.set()
        # Model/cloud resolution can outlive startup. Simulate completion when
        # shutdown has requested cancellation, without opening a late source.
        assert app.state.deployment._stopping.wait(5)
        resolved.set()
        return image_workflow()

    monkeypatch.setattr(app.state.workflow_definitions, "resolve", resolve)
    with TestClient(app) as client:
        assert entered.wait(1)
        assert not resolved.is_set()
        assert client.get("/healthz").status_code == 200
        deployment = client.get("/workflow-deployment").json()
        assert deployment["status"] == "starting"
        assert deployment["pipeline_id"] is None
    assert resolved.is_set()
    assert not opened
    assert app.state.deployment.status()["status"] == "stopped"


@pytest.mark.parametrize("value", ["[]", "null", "broken", [], {"value": float("nan")}])
def test_workflow_parameter_settings_require_a_json_object(tmp_path, value):
    with pytest.raises(ValueError, match="workflow_parameters"):
        EdgeSettings.from_app_config(
            {"workflow_parameters": value},
            model_root=tmp_path / "models",
            storage_root=tmp_path,
        )
