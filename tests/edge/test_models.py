"""RKNN contracts and lifecycle using a mocked platform session (no NPU)."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("INFERENCE_RUNTIME_PROFILE", "rv1126b")

from inference.edge import backend
from inference.edge.conversion import (
    ConversionRequest,
    ConversionResult,
    ConversionService,
)
from inference.edge.errors import EdgeError
from inference.edge.models import (
    EdgeModelManager,
    LocalModelStore,
    RegisteredModelStore,
)


def test_registered_models_require_real_backend_validation(package, kit, tmp_path):
    store = RegisteredModelStore(
        tmp_path / "bundled",
        [{"model_id": "detector/1", "directory": str(package.directory)}],
    )
    manager = EdgeModelManager(store)
    assert store.list_models() == []
    with pytest.raises(EdgeError, match="device validation"):
        manager.add_model("detector/1")
    store.validate(manager)
    assert store.validation["detector/1"]["status"] == "ready"
    assert store.list_models()[0]["model_id"] == "detector/1"
    assert len(kit.sessions) == 1 and kit.sessions[0].release_calls == 1
    kit.outputs = [np.zeros((1, 6, 7), dtype=np.float32)]
    store.validate(manager)
    assert store.validation["detector/1"]["status"] == "validation_failed"
    assert store.list_models() == []


@pytest.fixture
def package(tmp_path):
    directory = tmp_path / "detector" / "1"
    directory.mkdir(parents=True)
    blob = b"mock-rknn-model"
    (directory / "model.rknn").write_bytes(blob)
    metadata = {
        "schema_version": 1,
        "model_id": "detector/1",
        "platform": "rv1126b",
        "format": "rknn",
        "task": "object-detection",
        "model_file": "model.rknn",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "labels": ["person", "car"],
        "input": {
            "name": "images",
            "shape": [1, 32, 32, 3],
            "layout": "NHWC",
            "dtype": "uint8",
            "color_format": "RGB",
            "normalization": "baked",
        },
        "outputs": [
            {"name": "out", "shape": [1, 6, 8], "dtype": "float32", "layout": "BCN"}
        ],
        "postprocess": {
            "kind": "yolo-decoded",
            "box_format": "xywh",
            "scores": "probabilities",
        },
    }

    def write():
        (directory / "model.json").write_text(json.dumps(metadata))

    write()
    return SimpleNamespace(
        root=tmp_path,
        directory=directory,
        metadata=metadata,
        write=write,
        store=LocalModelStore(tmp_path),
    )


@pytest.fixture
def kit(monkeypatch):
    for name, value in {
        "RECAMERA_INFERENCE_SERVICE_SOCK": "/run/recamera/inferenced.sock",
        "RECAMERA_APP_ID": "inference",
        "RECAMERA_APP_INSTANCE": "test-instance",
        "RECAMERA_APP_GENERATION": "1",
    }.items():
        monkeypatch.setenv(name, value)
    state = SimpleNamespace(
        sessions=[],
        preprocess_calls=[],
        nms_calls=[],
        release_error=None,
        load_error=None,
        infer_error=None,
    )
    state.outputs = [np.zeros((1, 6, 8), dtype=np.float32)]
    # network center 16,16 / box 16x16 maps to original [4,4,12,12].
    state.outputs[0][0, :, 0] = [16, 16, 16, 16, 0.9, 0.1]
    state.outputs[0][0, :, 1] = [8, 8, 8, 8, 0.1, 0.8]

    class Session:
        def __init__(self, spec, **kwargs):
            if state.load_error:
                raise state.load_error
            self.spec, self.kwargs = spec, kwargs
            self.release_calls, self.inputs = 0, []
            state.sessions.append(self)

        def infer(self, inputs):
            self.inputs.append(inputs.copy())
            if state.infer_error:
                raise state.infer_error
            return [v.copy() for v in state.outputs]

        def release(self):
            self.release_calls += 1
            if state.release_error:
                raise state.release_error

    def preprocess(rgb, new_shape):
        state.preprocess_calls.append(rgb.copy())
        value = np.zeros((1, new_shape, new_shape, 3), dtype=np.uint8)
        info = SimpleNamespace(scale=2.0, pad_w=0, pad_h=0)
        return value, info

    def nms(boxes, scores, threshold):
        state.nms_calls.append((boxes.copy(), scores.copy(), threshold))
        return np.argsort(-scores).tolist()

    state.components = SimpleNamespace(
        session=Session,
        model_spec=lambda **v: SimpleNamespace(**v),
        tensor_spec=lambda **v: SimpleNamespace(**v),
        preprocess=preprocess,
        nms=nms,
        decode_dfl=lambda *a, **k: (
            np.zeros((0, 4)),
            np.zeros(0),
            np.zeros(0, dtype=int),
        ),
    )
    monkeypatch.setattr(backend, "_load_kit", lambda: state.components)
    return state


def test_store_checks_hash_and_does_not_modify_assets(package):
    before = (package.directory / "model.rknn").read_bytes()
    metadata = package.store.get("detector/1")
    assert metadata.path == package.directory / "model.rknn"
    assert metadata.input.shape == (1, 32, 32, 3)
    assert (package.directory / "model.rknn").read_bytes() == before
    (package.directory / "model.rknn").write_bytes(b"changed")
    with pytest.raises(EdgeError, match="SHA-256") as error:
        package.store.get("detector/1")
    assert error.value.code == "model_digest_mismatch"


@pytest.mark.parametrize(
    "patch",
    [
        {"platform": "rv1126"},
        {"format": "onnx"},
        {"model_file": "../../outside.rknn"},
        {"schema_version": True},
        {"labels": []},
        {"sha256": "a" * 63},
        {
            "input": {
                "name": "x",
                "shape": [1, 3, 32, 32],
                "layout": "NCHW",
                "dtype": "float32",
            }
        },
        {"postprocess": {"kind": "unknown", "scores": "probabilities"}},
        {
            "outputs": [
                {
                    "name": "out",
                    "shape": [1, 85, 100],
                    "layout": "BCN",
                    "dtype": "float32",
                }
            ]
        },
    ],
)
def test_rejects_incompatible_model_contracts(package, patch):
    package.metadata.update(patch)
    package.write()
    with pytest.raises(EdgeError) as error:
        package.store.get("detector/1")
    assert error.value.code == "invalid_model_metadata"


@pytest.mark.parametrize(
    "model_id", ["../detector/1", "/tmp/model", "a/../b", "a//b", ""]
)
def test_rejects_invalid_ids(package, model_id):
    with pytest.raises(EdgeError):
        package.store.get(model_id)


@pytest.mark.parametrize("logits", [False, True])
def test_objectness_multiplies_class_probability_before_threshold(package, kit, logits):
    package.metadata["outputs"][0]["shape"] = [1, 7, 8]
    package.metadata["postprocess"].update(
        objectness=True, scores="logits" if logits else "probabilities"
    )
    package.write()
    kit.outputs = [np.zeros((1, 7, 8), dtype=np.float32)]
    kit.outputs[0][0, :, 0] = [16, 16, 16, 16, 0.2, 0.9, 0.1]
    kit.outputs[0][0, :, 1] = [8, 8, 8, 8, 0.9, 0.1, 0.8]
    if logits:
        probabilities = np.clip(kit.outputs[0][:, 4:], 1e-6, 1 - 1e-6)
        kit.outputs[0][:, 4:] = np.log(probabilities / (1 - probabilities))
    model = backend.KitBackend(package.store.get("detector/1"))
    result = model.infer(np.zeros((16, 16, 3), dtype=np.uint8), confidence=0.25)
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["class"] == "car"
    assert prediction["confidence"] == pytest.approx(0.72)
    assert prediction["x"] == 4 and prediction["width"] == 4
    model.release()


def end2end_package(package, kit):
    package.metadata["outputs"] = [
        {"name": "out", "shape": [1, 3, 6], "dtype": "float32", "layout": "BNC"}
    ]
    package.metadata["postprocess"] = {
        "kind": "yolo-end2end",
        "box_format": "xyxy",
        "scores": "probabilities",
    }
    package.write()
    # Two overlapping instances of the same class must survive without a second NMS.
    kit.outputs = [
        np.array(
            [[[4, 4, 20, 20, 0.9, 0], [4, 4, 20, 20, 0.8, 0], [0, 0, 4, 4, 0.95, 1]]],
            dtype=np.float32,
        )
    ]


def test_end2end_preserves_overlaps_and_workflow_filter_limit_and_coordinates(
    package, kit
):
    end2end_package(package, kit)
    model = backend.KitBackend(package.store.get("detector/1"))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = model.infer(image, confidence=0.25, class_filter=["person"])
    assert [p["confidence"] for p in result["predictions"]] == pytest.approx([0.9, 0.8])
    assert {k: result["predictions"][0][k] for k in ("x", "y", "width", "height")} == {
        "x": 6,
        "y": 6,
        "width": 8,
        "height": 8,
    }
    assert kit.nms_calls == []
    assert len(model.infer(image, max_detections=1)["predictions"]) == 1
    assert model.infer(image, max_candidates=1)["predictions"][0]["class"] == "car"
    assert model.infer(image, confidence=0.99)["predictions"] == []
    model.release()


@pytest.mark.parametrize(
    "column,value",
    [(5, 1.5), (5, -1), (5, 2), (4, 1.1), (4, -0.1), (2, -1), (0, float("nan"))],
)
def test_end2end_rejects_invalid_classes_probabilities_and_boxes(
    package, kit, column, value
):
    end2end_package(package, kit)
    kit.outputs[0][0, 0, column] = value
    model = backend.KitBackend(package.store.get("detector/1"))
    with pytest.raises(EdgeError) as exc:
        model.infer(np.zeros((16, 16, 3), dtype=np.uint8))
    assert exc.value.code == "invalid_model_output"
    model.release()


@pytest.mark.parametrize("logits,nms", [(False, True), (True, False)])
def test_distance_head_uses_explicit_roles_grid_centers_and_nms_contract(
    package, kit, logits, nms
):
    package.metadata["outputs"] = [
        {
            "name": "boxes",
            "shape": [1, 4, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
            "role": "boxes",
        },
        {
            "name": "scores",
            "shape": [1, 2, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
            "role": "scores",
        },
    ]
    package.metadata["postprocess"] = {
        "kind": "yolo-distance",
        "scores": "logits" if logits else "probabilities",
        "nms": nms,
    }
    package.write()
    boxes = np.zeros((1, 4, 4, 4), dtype=np.float32)
    scores = np.full((1, 2, 4, 4), -8 if logits else 0, dtype=np.float32)
    boxes[0, :, 1, 2] = [0.5, 1, 0.5, 1]
    scores[0, 1, 1, 2] = 0 if logits else 0.9
    kit.outputs = [boxes, scores]
    model = backend.KitBackend(package.store.get("detector/1"))
    result = model.infer(np.zeros((16, 16, 3), dtype=np.uint8))
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["class"] == "car"
    assert prediction["confidence"] == pytest.approx(0.5 if logits else 0.9)
    assert {k: prediction[k] for k in ("x", "y", "width", "height")} == {
        "x": 10,
        "y": 6,
        "width": 4,
        "height": 8,
    }
    assert bool(kit.nms_calls) is nms
    model.release()


@pytest.mark.parametrize(
    "patch",
    [{"nms": True}, {"nms": "false"}, {"scores": "logits"}, {"box_format": "xywh"}],
)
def test_end2end_rejects_conflicting_metadata(package, kit, patch):
    end2end_package(package, kit)
    package.metadata["postprocess"].update(patch)
    package.write()
    with pytest.raises(EdgeError):
        package.store.get("detector/1")


@pytest.mark.parametrize("decoder", ["yolo-distance", "yolo-dfl"])
def test_raw_end2end_retains_secondary_classes_and_topk_before_filter(
    package, kit, decoder
):
    channels = 4 if decoder == "yolo-distance" else 64
    package.metadata["outputs"] = [
        {
            "name": "boxes",
            "shape": [1, channels, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
            "role": "boxes",
        },
        {
            "name": "scores",
            "shape": [1, 2, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
            "role": "scores",
        },
    ]
    package.metadata["postprocess"] = {
        "kind": decoder,
        "scores": "probabilities",
        "nms": False,
        "topk": 2,
        "reg_max": 16,
    }
    package.write()
    boxes = np.ones((1, channels, 4, 4), dtype=np.float32)
    if decoder == "yolo-dfl":
        boxes.fill(-80)
        boxes.reshape(1, 4, 16, 4, 4)[:, :, 1, :, :] = 80
    scores = np.zeros((1, 2, 4, 4), dtype=np.float32)
    scores[0, :, 1, 1] = [0.9, 0.8]
    scores[0, :, 2, 2] = [0.7, 0.6]
    kit.outputs = [boxes, scores]
    model = backend.KitBackend(package.store.get("detector/1"))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    predictions = model.infer(image)["predictions"]
    assert [p["class_id"] for p in predictions] == [0, 1]
    assert [p["confidence"] for p in predictions] == pytest.approx([0.9, 0.8])
    assert [(p["x"], p["y"], p["width"], p["height"]) for p in predictions] == [
        (6, 6, 8, 8)
    ] * 2
    filtered = model.infer(image, class_filter=["car"])["predictions"]
    assert len(filtered) == 1 and filtered[0]["confidence"] == pytest.approx(0.8)
    assert not kit.nms_calls
    model.release()


@pytest.mark.parametrize("topk", [True, 0, -1, 1001, "300"])
def test_rejects_unbounded_end2end_topk(package, topk):
    package.metadata["postprocess"]["topk"] = topk
    package.write()
    with pytest.raises(EdgeError):
        package.store.get("detector/1")


def test_rejects_asset_symlink_escape(package, tmp_path):
    target = tmp_path / "outside.rknn"
    target.write_bytes(b"outside")
    asset = package.directory / "model.rknn"
    asset.unlink()
    asset.symlink_to(target)
    with pytest.raises(EdgeError, match="inside its package"):
        package.store.get("detector/1")


def test_direct_launch_model_load_requires_appmgr(package, monkeypatch):
    for name in (
        "RECAMERA_INFERENCE_SERVICE_SOCK",
        "RECAMERA_INFERENCE_SERVICE",
        "RECAMERA_APP_GENERATION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        backend,
        "_load_kit",
        lambda: pytest.fail("Kit imported before authorization check"),
    )
    manager = EdgeModelManager(package.store)
    assert manager.list_models()[0]["loaded"] is False
    with pytest.raises(EdgeError) as error:
        manager.add_model("detector/1")
    assert error.value.code == "npu_authorization_required"
    assert error.value.status_code == 403


def test_session_contract_bgr_conversion_and_response(package, kit):
    manager = EdgeModelManager(package.store)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    result = manager.infer("detector/1", image)
    assert kit.preprocess_calls[0][0, 0].tolist() == [30, 20, 10]
    session = kit.sessions[0]
    assert session.kwargs["model_sha256"] == package.metadata["sha256"]
    assert session.kwargs["strict_inputs"] is True
    assert session.spec.inputs[0].name == "input"
    assert session.spec.inputs[0].shape == (1, 32, 32, 3)
    assert result["image"] == {"width": 16, "height": 16}
    assert result["predictions"][0]["x"] == 8
    assert result["predictions"][0]["width"] == 8
    assert result["predictions"][0]["class"] == "person"
    assert result.model_dump(by_alias=True) == result
    manager.clear()
    assert session.release_calls == 1
    assert (package.directory / "model.rknn").exists()


def test_filter_and_class_agnostic_option_reach_nms(package, kit):
    manager = EdgeModelManager(package.store)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = manager.infer("detector/1", image, class_filter=["car"], max_detections=1)
    assert [v["class"] for v in result["predictions"]] == ["car"]
    assert len(kit.nms_calls) == 1
    kit.nms_calls.clear()
    manager.infer("detector/1", image, class_agnostic_nms=True)
    assert len(kit.nms_calls) == 1
    assert len(kit.nms_calls[0][0]) == 2


def test_request_compatibility_serial_batch_and_single_session(package, kit):
    manager = EdgeModelManager(package.store)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    request = {
        "model_id": "detector/1",
        "image": [{"type": "numpy", "value": image}] * 2,
        "confidence": 0.85,
    }
    results = manager.infer_from_request_sync(request)
    assert len(results) == 2 and all(len(v["predictions"]) == 1 for v in results)
    assert len(kit.sessions) == 1 and len(kit.sessions[0].inputs) == 2
    manager.remove("other")
    assert kit.sessions[0].release_calls == 0
    manager.remove("detector/1")
    manager.clear()
    assert kit.sessions[0].release_calls == 1


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((0, 8, 3), dtype=np.uint8),
        "file.jpg",
    ],
)
def test_invalid_input_never_reaches_inference(package, kit, image):
    manager = EdgeModelManager(package.store)
    with pytest.raises(EdgeError) as error:
        manager.infer("detector/1", image)
    assert error.value.code == "invalid_input"
    assert not kit.sessions[0].inputs


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 6, 7), dtype=np.float32),
        np.zeros((1, 6, 8), dtype=np.float64),
        np.full((1, 6, 8), np.nan, dtype=np.float32),
        np.full((1, 6, 8), 2, dtype=np.float32),
    ],
)
def test_invalid_output_is_explicit(package, kit, output):
    kit.outputs = [output]
    with pytest.raises(EdgeError) as error:
        EdgeModelManager(package.store).infer(
            "detector/1", np.zeros((16, 16, 3), dtype=np.uint8)
        )
    assert error.value.code == "invalid_model_output"


def test_platform_permission_and_load_failure_leave_no_model(package, kit):
    manager = EdgeModelManager(package.store)
    kit.load_error = PermissionError("manifest did not authorize this model")
    with pytest.raises(EdgeError) as error:
        manager.add_model("detector/1")
    assert error.value.status_code == 403
    assert manager.describe_models() == []
    kit.load_error = RuntimeError("RKNN rejected incompatible binary")
    with pytest.raises(EdgeError) as error:
        manager.add_model("detector/1")
    assert error.value.code == "rknn_load_failed"
    assert manager.describe_models() == []


def test_release_failure_retains_handle_for_retry(package, kit):
    manager = EdgeModelManager(package.store)
    manager.add_model("detector/1")
    kit.release_error = OSError("service connection lost")
    with pytest.raises(EdgeError):
        manager.clear()
    assert len(manager.describe_models()) == 1
    with pytest.raises(EdgeError) as error:
        manager.infer("detector/1", np.zeros((16, 16, 3), dtype=np.uint8))
    assert error.value.code == "session_release_pending"
    kit.release_error = None
    manager.clear()
    assert kit.sessions[0].release_calls == 2
    assert manager.describe_models() == []


def test_model_switch_releases_first_before_loading_next(package, kit):
    import shutil

    second = package.root / "detector" / "2"
    shutil.copytree(package.directory, second)
    payload = dict(package.metadata, model_id="detector/2")
    (second / "model.json").write_text(json.dumps(payload))
    manager = EdgeModelManager(package.store)
    manager.add_model("detector/1")
    kit.release_error = OSError("failed unload")
    with pytest.raises(EdgeError):
        manager.add_model("detector/2")
    assert len(kit.sessions) == 1
    kit.release_error = None
    manager.add_model("detector/2")
    assert len(kit.sessions) == 2
    assert kit.sessions[0].release_calls == 2
    assert manager.describe_models()[0]["model_id"] == "detector/2"


def test_dfl_contract_and_empty_result(package, kit):
    package.metadata["postprocess"] = {
        "kind": "yolo-dfl",
        "reg_max": 16,
        "scores": "probabilities",
    }
    package.metadata["outputs"] = [
        {"name": "boxes", "shape": [1, 64, 4, 4], "dtype": "float32", "layout": "NCHW"},
        {"name": "scores", "shape": [1, 2, 4, 4], "dtype": "float32", "layout": "NCHW"},
    ]
    package.write()
    kit.outputs = [
        np.zeros((1, 64, 4, 4), dtype=np.float32),
        np.zeros((1, 2, 4, 4), dtype=np.float32),
    ]
    result = EdgeModelManager(package.store).infer(
        "detector/1", np.zeros((16, 16, 3), dtype=np.uint8)
    )
    assert result["predictions"] == []
    package.metadata["outputs"].pop()
    package.write()
    with pytest.raises(EdgeError, match="one box and one class"):
        package.store.get("detector/1")


def test_explicit_logits_are_sigmoided_even_when_values_fit_probability_range(
    package, kit
):
    package.metadata["postprocess"]["scores"] = "logits"
    package.write()
    kit.outputs[0].fill(0)
    kit.outputs[0][0, :, 0] = [16, 16, 8, 8, 0.5, 0.1]
    result = EdgeModelManager(package.store).infer(
        "detector/1",
        np.zeros((16, 16, 3), dtype=np.uint8),
        confidence=0.6,
    )
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["confidence"] == pytest.approx(
        1 / (1 + np.exp(-0.5))
    )


def test_dfl_logits_and_explicit_auxiliary_contract(package, kit):
    package.metadata["postprocess"] = {
        "kind": "yolo-dfl",
        "reg_max": 16,
        "scores": "logits",
    }
    package.metadata["outputs"] = [
        {
            "name": "boxes",
            "role": "boxes",
            "shape": [1, 64, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
        },
        {
            "name": "scores",
            "role": "scores",
            "shape": [1, 2, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
        },
        {
            "name": "sum",
            "role": "score_sum",
            "shape": [1, 1, 4, 4],
            "dtype": "float32",
            "layout": "NCHW",
        },
    ]
    package.write()
    kit.outputs = [
        np.zeros((1, 64, 4, 4), dtype=np.float32),
        np.full((1, 2, 4, 4), 0.5, dtype=np.float32),
        np.ones((1, 1, 4, 4), dtype=np.float32),
    ]
    seen = []

    def decode(outputs, *args, **kwargs):
        seen.extend(outputs)
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    kit.components.decode_dfl = decode
    result = EdgeModelManager(package.store).infer(
        "detector/1", np.zeros((16, 16, 3), dtype=np.uint8)
    )
    assert result["predictions"] == []
    assert len(seen) == 2
    np.testing.assert_allclose(seen[1], 1 / (1 + np.exp(-0.5)))
    # Neither the borrowed raw class logits nor auxiliary outputs were changed.
    assert (kit.outputs[1] == 0.5).all()
    package.metadata["outputs"][2].pop("role")
    package.write()
    with pytest.raises(EdgeError):
        package.store.get("detector/1")


def test_checked_in_example_matches_known_platform_model(tmp_path):
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "deploy/rv1126b/model.example.json"
    value = json.loads(source.read_text())
    directory = tmp_path / value["model_id"]
    directory.mkdir(parents=True)
    (directory / "model.json").write_text(json.dumps(value))
    metadata = LocalModelStore(tmp_path).get(value["model_id"], verify=False)
    assert (
        metadata.sha256
        == "5379721bb9b16fa3b9caceb8958175b2e8b0665d4ebcfe4e2524ad1db80d30da"
    )
    assert len(metadata.outputs) == 6
    assert metadata.score_format == "logits"
    assert sum(np.prod(v.shape) * 4 for v in metadata.outputs) == 4_838_400


def test_bad_preprocessing_does_not_reach_session(package, kit):
    kit.components.preprocess = lambda *a, **k: (
        np.zeros((1, 32, 32, 3), dtype=np.float32),
        None,
    )
    with pytest.raises(EdgeError) as error:
        EdgeModelManager(package.store).infer(
            "detector/1", np.zeros((16, 16, 3), dtype=np.uint8)
        )
    assert error.value.code == "invalid_preprocess_output"
    assert not kit.sessions[0].inputs


def test_inference_transport_failure_is_reported(package, kit):
    kit.infer_error = ConnectionError("daemon disconnected")
    with pytest.raises(EdgeError) as error:
        EdgeModelManager(package.store).infer(
            "detector/1", np.zeros((16, 16, 3), dtype=np.uint8)
        )
    assert error.value.code == "rknn_infer_failed"
    assert error.value.status_code == 503


def test_manager_serializes_concurrent_calls(package):
    state = SimpleNamespace(active=0, max_active=0, loads=0)

    class MockBackend:
        def __init__(self, metadata):
            state.loads += 1

        def infer(self, image, **kwargs):
            state.active += 1
            state.max_active = max(state.max_active, state.active)
            time.sleep(0.01)
            state.active -= 1
            return {"predictions": []}

        def release(self):
            pass

    manager = EdgeModelManager(package.store, backend_factory=MockBackend)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: manager.infer("detector/1", None), range(8)))
    assert state.max_active == 1 and state.loads == 1


def test_conversion_is_injected_and_never_assumes_network_protocol():
    service = ConversionService()
    assert service.status()["state"] == "unconfigured"
    with pytest.raises(EdgeError) as error:
        service.convert(ConversionRequest(source_reference="existing-model"))
    assert error.value.code == "converter_unconfigured"
    calls = []

    class Converter:
        def convert(self, request):
            calls.append(request)
            return ConversionResult(state="queued", job_reference="opaque-job-id")

    request = ConversionRequest("existing-model", options={"quantization": "int8"})
    service = ConversionService(Converter())
    assert service.configured
    assert service.convert(request).job_reference == "opaque-job-id"
    assert calls == [request]


def test_provider_padding_is_forwarded_to_preprocessing(package, kit):
    package.metadata["input"]["padding_value"] = 0
    package.write()
    calls = []

    def letterbox(rgb, new_shape, color):
        calls.append((new_shape, color))
        return np.full(
            (new_shape, new_shape, 3), color, dtype=np.uint8
        ), SimpleNamespace(scale=1, pad_w=0, pad_h=0)

    kit.components.letterbox = letterbox
    manager = EdgeModelManager(package.store)
    manager.infer("detector/1", np.full((16, 32, 3), 255, dtype=np.uint8))
    assert calls == [(32, 0)]
    assert not kit.sessions[0].inputs[0].any()
    assert kit.preprocess_calls == []
    manager.clear()
    package.metadata["input"]["padding_value"] = -1
    package.write()
    with pytest.raises(EdgeError, match="padding"):
        package.store.get("detector/1")
