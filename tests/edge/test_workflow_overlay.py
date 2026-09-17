import base64
import os

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

from inference.edge.overlay import MAX_DETECTIONS, extract_overlays
from inference.edge.preview import WorkflowPreview


def detection(**changes):
    return {
        "x": 50,
        "y": 25,
        "width": 40,
        "height": 20,
        "confidence": 0.9,
        "class": "person",
        **changes,
    }


def output(predictions=None, width=100, height=50):
    return {
        "image": {"width": width, "height": height},
        "predictions": [detection()] if predictions is None else predictions,
    }


def test_root_coordinates_bounds_and_own_crop_are_not_confused():
    groups = extract_overlays(
        [
            {
                "model": output(),
                "crop": output(width=25),
                "own": output(),
                "image": {"type": "base64", "value": "ignored"},
            }
        ],
        (100, 50),
        {"model", "crop", "image"},
    )
    assert len(groups) == 1
    group = next(iter(groups.values()))
    assert group["name"] == "model"
    assert group["results"][0]["box"] == [0.3, 0.3, 0.7, 0.7]
    assert group["results"][0]["spaces"] == {"box": "normalized_xyxy"}
    assert "base64" not in str(groups)
    invalid = [detection(x=float("nan")), detection(width=-1), detection(x=200)]
    groups = extract_overlays({"model": output(invalid)}, (100, 50))
    assert next(iter(groups.values()))["results"] == []


def test_empty_output_is_drawable_and_total_detection_budget_is_bounded():
    empty = extract_overlays({"model": output([], None, None)}, (100, 50))
    assert next(iter(empty.values()))["results"] == []
    many = extract_overlays(
        [{str(i): output([detection()] * 150) for i in range(30)}], (100, 50)
    )
    assert len(many) == 16
    assert sum(len(g["results"]) for g in many.values()) == MAX_DETECTIONS


def test_builder_own_root_output_is_accepted_but_resized_crop_is_rejected():
    groups = extract_overlays(
        {
            "root": output([detection(parent_id="camera")]),
            "crop": output([detection(parent_id="crop-uuid")]),
            "other_input": output([detection(parent_id="image")]),
            "mixed": output(
                [detection(parent_id="camera"), detection(parent_id="crop")]
            ),
        },
        (100, 50),
        root_outputs=set(),
        image_input="camera",
    )
    assert [group["name"] for group in groups.values()] == ["root"]
    assert next(iter(groups.values()))["results"][0]["box"] == [0.3, 0.3, 0.7, 0.7]


def test_builder_own_empty_frame_clears_boxes_without_losing_known_output():
    preview = WorkflowPreview(1024)
    preview.reset("a" * 32)
    record = {"pipeline_id": "a" * 32, "frame_id": 1, "frame_timestamp": "now"}
    preview.publish(
        {**record, "result": {"model": output([detection(parent_id="image")])}},
        None,
        (100, 50),
        set(),
        "image",
    )
    assert len(preview.overlay()["groups"]) == 1
    preview.publish(
        {**record, "frame_id": 2, "result": {"model": output([])}},
        None,
        (100, 50),
        set(),
        "image",
    )
    assert preview.overlay()["groups"] == []
    assert len(preview.status()["overlay_outputs"]) == 1


def test_rtsp_original_and_overlay_always_share_one_frame_and_clear_on_missing_result(
    monkeypatch,
):
    preview = WorkflowPreview(1024)
    pipeline = "a" * 32
    preview.reset(pipeline)
    preview.overlay(include_image=True)

    def publish(index, result, image):
        preview.publish(
            {
                "pipeline_id": pipeline,
                "frame_id": index,
                "frame_timestamp": str(index),
                "result": result,
            },
            image,
            (100, 50),
        )

    publish(1, {"model": output()}, b"original-1")
    metadata = preview.status()
    assert metadata["outputs"] == [] and len(metadata["overlay_outputs"]) == 1
    plain = preview.overlay()
    assert "image" not in plain and plain["frame_id"] == 1
    paired = preview.overlay(include_image=True)
    assert base64.b64decode(paired["image"]) == b"original-1"
    assert paired["groups"][0]["results"][0]["box"] == [0.3, 0.3, 0.7, 0.7]
    publish(2, {"model": None}, b"original-2")
    assert preview.overlay(include_image=True)["groups"] == []
    assert (
        base64.b64decode(preview.overlay(include_image=True)["image"]) == b"original-2"
    )
    assert len(preview.status()["overlay_outputs"]) == 1
    monkeypatch.setattr(
        "inference.edge.preview.time.monotonic", lambda: preview._updated + 20
    )
    assert preview.overlay() is None
    preview.reset("b" * 32)
    assert preview.status()["overlay_outputs"] == []
