"""Workflow draft persistence and the original Builder's lightweight contracts."""

import hashlib
import json
import os

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from inference.edge.builder import create_builder_router
from inference.edge.errors import EdgeError
from inference.edge.limits import encode_bounded_json
from inference.edge.settings import EdgeSettings
from inference.edge.storage import WorkflowStore
from inference.edge.workflows import EdgeWorkflows


def minimal_workflow():
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "detect",
                "images": "$inputs.image",
                "model_id": "detector/1",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "objects",
                "selector": "$steps.detect.predictions",
            }
        ],
    }


class NoInferenceManager:
    def list_models(self):
        return [
            {
                "model_id": "detector/1",
                "format": "rknn",
                "task_type": "object-detection",
                "loaded": False,
            },
            {
                "model_id": "unsupported/1",
                "format": "onnx",
                "task_type": "object-detection",
                "loaded": False,
            },
        ]

    def add_model(self, *args, **kwargs):
        raise AssertionError("Editor endpoints must not load an NPU model")

    def infer_from_request_sync(self, *args, **kwargs):
        raise AssertionError("Editor endpoints must not infer")


def test_drafts_survive_recreation_and_preserve_builder_contract(tmp_path):
    store = WorkflowStore(tmp_path)
    draft = {
        "name": "门口检测",
        "specification": minimal_workflow(),
        "edge_ui": {"positions": {"detect": {"x": 10, "y": 20}}},
    }
    store.save("door", draft)
    stored = WorkflowStore(tmp_path).get("door")
    assert stored["data"]["config"] == {**draft, "id": "door"}
    assert isinstance(stored["data"]["createTime"], int)
    listed = store.list()["data"]["door"]
    assert isinstance(listed["updateTime"]["_seconds"], int)
    assert listed["config"] == stored["data"]["config"]
    store.save("renamed", {**draft, "id": "door"})
    assert set(store.list()["data"]) == {"renamed"}
    store.delete("renamed")
    assert store.list() == {"data": {}}


def test_rename_cannot_destroy_source_after_write_failure_or_target_collision(
    tmp_path, monkeypatch
):
    store = WorkflowStore(tmp_path)
    store.save("old", {"name": "Original"})
    store.save("existing", {"name": "Existing"})
    with pytest.raises(EdgeError) as collision:
        store.save("existing", {"id": "old", "name": "Renamed"})
    assert collision.value.status_code == 409
    monkeypatch.setattr(
        os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure"))
    )
    with pytest.raises(OSError):
        store.save("new", {"id": "old", "name": "Renamed"})
    assert set(store.list()["data"]) == {"old", "existing"}
    assert store.get("old")["data"]["config"]["name"] == "Original"
    assert not list(tmp_path.glob(".workflow-*.tmp"))


def test_storage_limits_preserve_existing_drafts_and_allow_rename_at_capacity(tmp_path):
    store = WorkflowStore(
        tmp_path, max_workflows=1, max_file_bytes=128, max_total_bytes=128
    )
    store.save("first", {"data": "ok"})
    for identifier, body in (("second", {}), ("first", {"data": "x" * 128})):
        with pytest.raises(EdgeError) as error:
            store.save(identifier, body)
        assert error.value.code == "workflow_storage_limit"
        assert error.value.status_code == 413
    store.save("renamed", {"id": "first", "data": "kept"})
    assert set(store.list()["data"]) == {"renamed"}
    assert store.get("renamed")["data"]["config"]["data"] == "kept"
    aggregate = WorkflowStore(
        tmp_path / "aggregate", max_workflows=3, max_file_bytes=128, max_total_bytes=64
    )
    aggregate.save("first", {"data": "x" * 20})
    with pytest.raises(EdgeError) as error:
        aggregate.save("second", {"data": "x" * 20})
    assert error.value.code == "workflow_storage_limit"
    assert set(aggregate.list()["data"]) == {"first"}


def test_storage_rejects_escape_symlinks_and_corrupt_ids(tmp_path):
    store = WorkflowStore(tmp_path / "drafts")
    for identifier in ("../escape", "models", "", "x" * 129):
        with pytest.raises(EdgeError) as error:
            store.save(identifier, {})
        assert error.value.code == "invalid_workflow_id"
    target = tmp_path / "outside.json"
    target.write_text('{"id":"safe"}')
    hashed = store.root / (hashlib.sha256(b"safe").hexdigest() + ".json")
    hashed.symlink_to(target)
    with pytest.raises(EdgeError) as error:
        store.get("safe")
    assert error.value.code == "workflow_storage_unavailable"
    assert target.read_text() == '{"id":"safe"}'
    hashed.unlink()
    hashed.write_text('{"id":"different"}')
    with pytest.raises(EdgeError) as error:
        store.list()
    assert error.value.code == "workflow_storage_invalid"


def test_builder_crud_schema_validation_and_dynamic_outputs_without_npu(tmp_path):
    manager = NoInferenceManager()
    settings = EdgeSettings()
    workflows = EdgeWorkflows(manager, settings)
    executed = []

    async def execute(function, *args):
        executed.append(function.__name__)
        return function(*args)

    app = FastAPI()

    @app.exception_handler(EdgeError)
    async def edge_error(request, exc):
        return JSONResponse({"error": exc.code}, status_code=exc.status_code)

    app.include_router(
        create_builder_router(
            workflows, manager, WorkflowStore(tmp_path), execute=execute
        )
    )
    with TestClient(app) as client:
        draft = {"name": "Draft", "specification": minimal_workflow()}
        assert client.post("/build/api/test", json=draft).status_code == 201
        assert (
            client.get("/build/api").json()["data"]["test"]["config"]["name"] == "Draft"
        )
        assert (
            client.get("/build/api/test").json()["data"]["config"]["specification"]
            == minimal_workflow()
        )
        models = client.get("/build/api/models").json()["models"]
        assert [model["model_id"] for model in models] == [
            "detector/1",
            "yolov8n-640",
            "yolov11n-640",
            "yolo26n-640",
        ]
        assert models[1]["preparation_required"] is True
        for version in (1, 2, 3):
            assert (
                f"roboflow_core/roboflow_object_detection_model@v{version}"
                in models[0]["compatible_block_types"]
            )
        schema = client.get("/workflows/definition/schema")
        assert schema.status_code == 200
        assert "ObjectDetectionModel" in json.dumps(schema.json()["schema"])
        assert client.post("/workflows/validate", json=minimal_workflow()).json() == {
            "status": "ok"
        }
        invalid = minimal_workflow()
        invalid["steps"][0]["type"] = "roboflow_core/clip@v1"
        assert client.post("/workflows/validate", json=invalid).status_code == 422
        step = minimal_workflow()["steps"][0]
        outputs = client.post("/workflows/blocks/dynamic_outputs", json=step)
        assert outputs.status_code == 200
        assert [output["name"] for output in outputs.json()] == [
            "inference_id",
            "predictions",
        ]
        assert (
            client.post(
                "/workflows/blocks/dynamic_outputs", json=invalid["steps"][0]
            ).status_code
            == 422
        )
        assert client.delete("/build/api/test").status_code == 200
        assert client.get("/build/api/test").status_code == 404
    assert {"schema", "validate", "dynamic_outputs", "save", "delete"} <= set(executed)
    described = workflows.describe()
    assert (
        len(encode_bounded_json(described, settings.max_response_bytes))
        < settings.max_response_bytes
    )
    assert described["kinds_connections"]
    assert described["primitives_connections"]
    assert described["universal_query_language_description"]["operations_description"]
    assert described["dynamic_block_definition_schema"]["not"] == {}
