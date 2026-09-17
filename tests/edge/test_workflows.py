"""Execute real Workflow graphs in a process that forbids ML framework imports."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BOOTSTRAP = """
import importlib.abc
import sys

class NoTrainingFrameworks(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "torchvision", "onnxruntime", "inference_models",
            "transformers", "diffusers",
        }:
            self.attempted.append(fullname)
            raise AssertionError("Forbidden device import: " + fullname)

sys.meta_path.insert(0, NoTrainingFrameworks())
import numpy as np
from inference.core.workflows.execution_engine.core import ExecutionEngine
"""


def run_edge_script(code):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BOOTSTRAP
            + textwrap.dedent(code)
            + "\nassert not NoTrainingFrameworks.attempted, NoTrainingFrameworks.attempted",
        ],
        cwd=ROOT,
        env={**os.environ, "INFERENCE_RUNTIME_PROFILE": "rv1126b"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_original_engine_executes_numpy_static_crop_without_frameworks():
    run_edge_script(
        """
        workflow = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [{
                "type": "AbsoluteStaticCrop", "name": "crop",
                "image": "$inputs.image", "x_center": 6, "y_center": 5,
                "width": 4, "height": 2,
            }],
            "outputs": [{"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}],
        }
        image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        engine = ExecutionEngine.init(workflow_definition=workflow)
        result = engine.run(runtime_parameters={"image": image})
        np.testing.assert_array_equal(result[0]["crop"].numpy_image, image[4:6, 4:8])
        assert result[0]["crop"].numpy_image.dtype == np.uint8
        assert result[0]["crop"]._tensor_image is None
    """
    )


def test_detection_filter_and_dynamic_crop_preserve_original_graph_semantics():
    run_edge_script(
        """
        class ModelManager:
            def __init__(self):
                self.calls = []
            def infer_from_request_sync(self, model_id, request):
                image = request["image"]["value"]
                self.calls.append((model_id, image.copy()))
                assert image.dtype == np.uint8
                return {
                    "inference_id": "test-inference",
                    "image": {"width": image.shape[1], "height": image.shape[0]},
                    "predictions": [
                        {"x": 4., "y": 3., "width": 4., "height": 2.,
                         "confidence": .9, "class": "person", "class_id": 0},
                        {"x": 4., "y": 3., "width": 4., "height": 2.,
                         "confidence": .4, "class": "person", "class_id": 0},
                        {"x": 4., "y": 3., "width": 4., "height": 2.,
                         "confidence": .95, "class": "car", "class_id": 1},
                    ],
                }
        def statement(property_name, comparator, value):
            return {
                "type": "BinaryStatement",
                "left_operand": {"type": "DynamicOperand", "operations": [
                    {"type": "ExtractDetectionProperty", "property_name": property_name}
                ]},
                "comparator": {"type": comparator},
                "right_operand": {"type": "StaticOperand", "value": value},
            }
        workflow = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {"type": "AbsoluteStaticCrop", "name": "roi", "image": "$inputs.image",
                 "x_center": 10, "y_center": 8, "width": 12, "height": 10},
                {"type": "ObjectDetectionModel", "name": "detect", "image": "$steps.roi.crops",
                 "model_id": "detector/1", "confidence": .3},
                {"type": "DetectionsFilter", "name": "filter", "predictions": "$steps.detect.predictions",
                 "operations": [{"type": "DetectionsFilter", "filter_operation": {
                     "type": "StatementGroup", "operator": "and", "statements": [
                         statement("class_name", "in (Sequence)", ["person"]),
                         statement("confidence", "(Number) >=", .8),
                     ],
                 }}]},
                {"type": "PropertyDefinition", "name": "count", "data": "$steps.filter.predictions",
                 "operations": [{"type": "SequenceLength"}]},
                {"type": "DynamicCrop", "name": "objects", "images": "$steps.roi.crops",
                 "predictions": "$steps.filter.predictions"},
            ],
            "outputs": [
                {"type": "JsonField", "name": "predictions", "selector": "$steps.filter.predictions"},
                {"type": "JsonField", "name": "own", "selector": "$steps.filter.predictions", "coordinates_system": "own"},
                {"type": "JsonField", "name": "crops", "selector": "$steps.objects.crops"},
                {"type": "JsonField", "name": "count", "selector": "$steps.count.output"},
            ],
        }
        manager = ModelManager()
        engine = ExecutionEngine.init(workflow_definition=workflow, init_parameters={"model_manager": manager})
        image = np.arange(20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3)
        result = engine.run(runtime_parameters={"image": image})[0]
        assert len(manager.calls) == 1
        np.testing.assert_array_equal(manager.calls[0][1], image[3:13, 4:16])
        assert result["count"] == 1
        assert len(result["predictions"]) == 1
        np.testing.assert_array_equal(result["predictions"].xyxy, [[6, 5, 10, 7]])
        np.testing.assert_array_equal(result["own"].xyxy, [[2, 2, 6, 4]])
        assert len(result["crops"]) == 1
        np.testing.assert_array_equal(result["crops"][0].numpy_image, image[5:7, 6:10])
        serialized = engine.run(runtime_parameters={"image": image}, serialize_results=True)[0]
        assert serialized["predictions"]["predictions"][0]["class"] == "person"
        assert serialized["predictions"]["predictions"][0]["x"] == 8
    """
    )


def test_unsupported_model_block_fails_before_initializing_resources():
    run_edge_script(
        """
        from inference.core.workflows.errors import WorkflowError
        workflow = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [{"type": "roboflow_core/clip@v1", "name": "clip", "images": "$inputs.image"}],
            "outputs": [],
        }
        try:
            ExecutionEngine.init(workflow_definition=workflow)
        except WorkflowError:
            pass
        else:
            raise AssertionError("Unsupported model block unexpectedly compiled")
    """
    )


def test_builder_metadata_and_schema_do_not_import_training_frameworks():
    run_edge_script(
        """
        from inference.edge.settings import EdgeSettings
        from inference.edge.workflows import EdgeWorkflows
        from inference.edge.limits import encode_bounded_json
        from inference.edge.errors import EdgeError
        settings = EdgeSettings()
        workflows = EdgeWorkflows(object(), settings)
        described = workflows.describe()
        assert len(described['blocks']) >= 100
        assert described['kinds_connections']
        assert described['primitives_connections']
        assert described['universal_query_language_description']['operations_description']
        assert len(encode_bounded_json(described, 16 * 1024 * 1024)) < 16 * 1024 * 1024
        assert 'schema' in workflows.schema()
        outputs = workflows.dynamic_outputs({
            'type': 'PropertyDefinition', 'name': 'count',
            'data': '$steps.detect.predictions', 'operations': [{'type': 'SequenceLength'}],
        })
        assert [value['name'] for value in outputs] == ['output']
        try:
            workflows.describe(dynamic_blocks_definitions=[{'type': 'custom'}])
        except EdgeError as exc:
            assert exc.code == 'unsupported_workflow_block'
        else:
            raise AssertionError('Dynamic Python definitions were not rejected')
        """
    )


def test_continue_if_preserves_branch_selection():
    run_edge_script(
        """
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {"type": "WorkflowParameter", "name": "enabled"},
            ],
            "steps": [
                {"type": "ContinueIf", "name": "gate",
                 "stop_delay": 0,
                 "condition_statement": {"type": "StatementGroup", "statements": [{
                     "type": "UnaryStatement", "operand": {"type": "DynamicOperand", "operand_name": "enabled"},
                     "operator": {"type": "(Boolean) is True"},
                 }]},
                 "evaluation_parameters": {"enabled": "$inputs.enabled"},
                 "next_steps": ["$steps.crop"]},
                {"type": "RelativeStaticCrop", "name": "crop", "images": "$inputs.image",
                 "x_center": .5, "y_center": .5, "width": .5, "height": .5},
            ],
            "outputs": [{"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}],
        }
        engine = ExecutionEngine.init(workflow_definition=workflow)
        image = np.zeros((20, 24, 3), dtype=np.uint8)
        enabled = engine.run(runtime_parameters={"image": image, "enabled": True})
        assert enabled[0]["crop"].numpy_image.shape == (10, 12, 3)
        disabled = engine.run(runtime_parameters={"image": image, "enabled": False})
        assert disabled[0]["crop"] is None
    """
    )


def test_http_workflow_errors_and_image_output_are_normalized():
    run_edge_script(
        """
        import base64
        import cv2
        from fastapi.testclient import TestClient
        from inference.edge.api import create_app
        from inference.edge.errors import EdgeError

        class Manager:
            def clear(self):
                pass
            def infer_from_request_sync(self, **kwargs):
                raise EdgeError("NPU busy", code="npu_unavailable", status_code=503)

        specification = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [{"type": "AbsoluteStaticCrop", "name": "crop", "image": "$inputs.image",
                       "x_center": 10, "y_center": 10, "width": 8, "height": 6}],
            "outputs": [{"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}],
        }
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        inputs = {"image": {"type": "base64", "value": base64.b64encode(encoded).decode()}}
        with TestClient(create_app(manager=Manager())) as client:
            response = client.post("/workflows/run", json={"specification": specification, "inputs": inputs})
            assert response.status_code == 200, response.text
            crop = response.json()["outputs"][0]["crop"]
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(crop["value"]), np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (6, 8, 3)
            response = client.post("/workflows/run", json={"specification": specification, "inputs": {}})
            assert response.status_code == 400, response.text
            assert response.json()["error"] == "invalid_workflow"
            for bad in [
                {**specification, "steps": None},
                {**specification, "steps": [None]},
                {**specification, "dynamic_blocks_definitions": [{"name": "custom"}]},
                {**specification, "steps": [{"type": "unavailable_block", "name": "unsupported"}]},
            ]:
                response = client.post("/workflows/run", json={"specification": bad, "inputs": inputs})
                assert response.status_code == 422, response.text
            detection = {
                **specification,
                "steps": [{"type": "rv1126b/rknn_object_detection@v1", "name": "detect", "image": "$inputs.image", "model_id": "detector/1"}],
                "outputs": [{"type": "JsonField", "name": "predictions", "selector": "$steps.detect.predictions"}],
            }
            response = client.post("/workflows/run", json={"specification": detection, "inputs": inputs})
            assert response.status_code == 503, response.text
            assert response.json()["error"] == "npu_unavailable"
    """
    )


def test_video_metadata_is_preserved_and_workflow_caches_are_bounded():
    run_edge_script(
        """
        from datetime import datetime, timezone
        from inference.edge.settings import EdgeSettings
        from inference.edge.workflows import EdgeWorkflows
        from inference.core.workflows.execution_engine.v1.compiler.core import COMPILATION_CACHE
        from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import WORKFLOW_DEFINITION_ENTITIES_CACHE
        workflow = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [{"type": "PropertyDefinition", "name": "frame", "data": "$inputs.image",
                       "operations": [{"type": "ExtractFrameMetadata", "property_name": "frame_number"}]}],
            "outputs": [{"type": "JsonField", "name": "frame", "selector": "$steps.frame.output"}],
        }
        runner = EdgeWorkflows(object(), EdgeSettings(workflow_cache_size=2))
        inputs = {"image": np.zeros((8, 8, 3), np.uint8)}
        result = runner.run(workflow, inputs, stream_id="video", frame_metadata={
            "frame_number": 17, "frame_timestamp": datetime.now(timezone.utc), "fps": 10,
        })
        assert result[0]["frame"] == 17
        assert runner.run(workflow, inputs, stream_id="video")[0]["frame"] == 18
        for index in range(6):
            step_name = "frame" + str(index)
            graph = {**workflow, "steps": [{**workflow["steps"][0], "name": step_name}],
                     "outputs": [{"type": "JsonField", "name": "frame", "selector": "$steps." + step_name + ".output"}]}
            assert runner.run(graph, inputs, stream_id="video" + str(index))[0]["frame"] == 1
        assert len(runner._cache) == 2
        assert len(runner._frame_numbers) == 2
        assert len(COMPILATION_CACHE._cache) <= 4
        assert len(WORKFLOW_DEFINITION_ENTITIES_CACHE._cache) <= 2
        runner.clear()
        assert not runner._cache and not runner._frame_numbers
    """
    )
