"""Project structured Workflow outputs onto the original input coordinate space."""

import hashlib
import json
import math

MAX_OUTPUTS = 16
MAX_DETECTIONS = 128


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def extract_overlays(result, size, root_outputs=None, image_input=None):
    """Only accept explicit detections already exported in root coordinates.

    Never infer coordinates from rendered pixels or silently stretch crop-space
    detections onto the source. Empty detections still declare a drawable output.
    """
    width, height = size
    if not width or not height:
        return {}
    groups = {}
    batches = [result] if isinstance(result, dict) else result
    if not isinstance(batches, list):
        return groups
    remaining = MAX_DETECTIONS

    def visit(value, path, name, root_coordinates, depth=0):
        nonlocal remaining
        if len(groups) >= MAX_OUTPUTS or depth > 5:
            return
        if isinstance(value, list):
            for index, item in enumerate(value[:MAX_OUTPUTS]):
                visit(
                    item,
                    [*path, index],
                    f"{name}[{index}]",
                    root_coordinates,
                    depth + 1,
                )
            return
        if not isinstance(value, dict) or not isinstance(
            value.get("predictions"), list
        ):
            return
        image = value.get("image")
        if not isinstance(image, dict):
            return
        predictions = value["predictions"]
        if predictions and (image.get("width"), image.get("height")) != size:
            return
        # The builder defaults to "own", including for models on the root
        # input. Such boxes are safe only when their parent is that input.
        # A resized crop can have identical dimensions, so size alone cannot
        # establish the coordinate space. Empty own-space results cannot prove
        # their parent; previously discovered outputs remain in preview metadata.
        if not root_coordinates and (
            not image_input
            or not predictions
            or not all(
                isinstance(prediction, dict)
                and prediction.get("parent_id") == image_input
                for prediction in predictions
            )
        ):
            return
        entries = []
        for prediction in predictions[:remaining]:
            if not isinstance(prediction, dict):
                continue
            box = [prediction.get(key) for key in ("x", "y", "width", "height")]
            if not all(_number(v) for v in box) or box[2] <= 0 or box[3] <= 0:
                continue
            x, y, w, h = box
            coords = [
                (x - w / 2) / width,
                (y - h / 2) / height,
                (x + w / 2) / width,
                (y + h / 2) / height,
            ]
            coords = [max(0, min(1, v)) for v in coords]
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            entry = {
                "box": coords,
                "label": str(prediction.get("class", ""))[:160],
                "spaces": {"box": "normalized_xyxy"},
            }
            score = prediction.get("confidence")
            if _number(score):
                entry["score"] = max(0, min(1, score))
            entries.append(entry)
            remaining -= 1
        identifier = hashlib.sha256(json.dumps(path).encode()).hexdigest()[:20]
        groups[identifier] = {"id": identifier, "name": name[:256], "results": entries}

    for batch_index, batch in enumerate(batches[:MAX_OUTPUTS]):
        if not isinstance(batch, dict):
            continue
        for name, value in list(batch.items())[:128]:
            visit(
                value,
                [batch_index, name],
                str(name),
                root_outputs is None or name in root_outputs,
            )
    return groups
