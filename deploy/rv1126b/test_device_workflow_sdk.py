#!/usr/bin/env python3
"""HTTP-only acceptance of an already running RV1126B app with its official SDK.

No camera/RTC lifecycle calls, SSH, installation, platform grants or RKNN imports.
The only device writes are fresh, uniquely named test Workflows and their cleanup.
Use --prepare-only to validate inputs and write specs without any network request.
Run sequentially, only after the device test coordinator reserves the idle app.
"""

import argparse
import base64
import copy
import hashlib
import inspect
import json
import os
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)
from uuid import uuid4


def check(condition, reason):
    if not condition:
        raise AssertionError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def graph(model_id, version):
    detector = {
        "type": f"roboflow_core/roboflow_object_detection_model@v{version}",
        "name": "detect",
        "images": "$inputs.image",
        "model_id": model_id,
        "iou_threshold": 0.45,
        "max_detections": 100,
        "class_filter": "$inputs.classes",
    }
    if version == 3:
        detector.update(
            confidence_mode="custom", custom_confidence="$inputs.confidence"
        )
    else:
        detector["confidence"] = "$inputs.confidence"
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.25},
            {"type": "WorkflowParameter", "name": "classes", "default_value": []},
        ],
        "steps": [
            detector,
            {
                "type": "PropertyDefinition",
                "name": "count",
                "data": "$steps.detect.predictions",
                "operations": [{"type": "SequenceLength"}],
            },
            {
                "type": "BoundingBoxVisualization",
                "name": "boxes",
                "image": "$inputs.image",
                "predictions": "$steps.detect.predictions",
            },
            {
                "type": "roboflow_core/label_visualization@v2",
                "name": "labels",
                "image": "$steps.boxes.image",
                "predictions": "$steps.detect.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detect.predictions",
            },
            {"type": "JsonField", "name": "count", "selector": "$steps.count.output"},
            {"type": "JsonField", "name": "image", "selector": "$steps.labels.image"},
            {"type": "JsonField", "name": "plain", "selector": "$inputs.image"},
        ],
    }


def filtered_graph(base, class_name, confidence):
    spec = copy.deepcopy(base)

    def statement(name, comparator, value):
        return {
            "type": "BinaryStatement",
            "left_operand": {
                "type": "DynamicOperand",
                "operations": [
                    {"type": "ExtractDetectionProperty", "property_name": name}
                ],
            },
            "comparator": {"type": comparator},
            "right_operand": {"type": "StaticOperand", "value": value},
        }

    spec["steps"].insert(
        1,
        {
            "type": "DetectionsFilter",
            "name": "filtered",
            "predictions": "$steps.detect.predictions",
            "operations": [
                {
                    "type": "DetectionsFilter",
                    "filter_operation": {
                        "type": "StatementGroup",
                        "operator": "and",
                        "statements": [
                            statement("class_name", "in (Sequence)", [class_name]),
                            statement("confidence", "(Number) >=", confidence),
                        ],
                    },
                }
            ],
        },
    )
    for step in spec["steps"][2:]:
        for key in ("data", "predictions"):
            if step.get(key) == "$steps.detect.predictions":
                step[key] = "$steps.filtered.predictions"
    spec["outputs"][0]["selector"] = "$steps.filtered.predictions"
    return spec


def nested_graph(child_id):
    spec = graph("unused", 3)
    spec["steps"] = [
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "inner",
            "workflow_workspace_id": "local",
            "workflow_id": child_id,
            "parameter_bindings": {
                "image": "$inputs.image",
                "confidence": "$inputs.confidence",
                "classes": "$inputs.classes",
            },
        }
    ]
    spec["outputs"] = [
        {"type": "JsonField", "name": name, "selector": f"$steps.inner.{name}"}
        for name in ("predictions", "count", "image", "plain")
    ]
    return spec


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class API:
    def __init__(self, origin, token, report):
        self.origin, self.token, self.report = origin, token, report
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, method, path, body=None, expected=200):
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept": "application/json",
        }
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        event = {"method": method, "path": path}
        self.report["http"].append(event)
        start = time.monotonic()
        try:
            try:
                response = self.opener.open(
                    Request(
                        self.origin + path, data=payload, headers=headers, method=method
                    ),
                    timeout=90,
                )
            except HTTPError as exc:
                response = exc
            with response:
                event["status"] = response.code
                raw = response.read(16 * 1024 * 1024 + 1)
            check(len(raw) <= 16 * 1024 * 1024, "HTTP response exceeded test budget")
            value = json.loads(raw)
            check(
                event["status"] == expected,
                f"Unexpected HTTP status at {method} {path}: {event['status']}",
            )
            return value
        finally:
            event["elapsed_ms"] = round((time.monotonic() - start) * 1000, 2)


@contextmanager
def deadline(seconds):
    def timed_out(*_):
        raise TimeoutError("Official SDK request exceeded its acceptance-test deadline")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def decode_visual(value, cv2, np):
    if isinstance(value, np.ndarray):
        return value, {"format": "numpy", "sha256": digest(value.tobytes())}
    if isinstance(value, dict):
        value = value.get("value")
    check(isinstance(value, str), "SDK visualization is neither base64 nor NumPy")
    raw = base64.b64decode(value.split(";base64,", 1)[-1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    check(image is not None, "Workflow visualization is not a decodable image")
    return image, {
        "format": "encoded",
        "encoded_bytes": len(raw),
        "sha256": digest(raw),
    }


def detection_list(value):
    if isinstance(value, dict):
        value = value.get("predictions")
    check(
        isinstance(value, list),
        "Predictions do not follow the original detection JSON schema",
    )
    return [
        {
            key: row[key]
            for key in ("x", "y", "width", "height", "confidence", "class", "class_id")
            if key in row
        }
        for row in value
    ]


def canonical(rows):
    return sorted(
        tuple(
            round(float(row[key]), 5)
            for key in ("x", "y", "width", "height", "confidence")
        )
        + (str(row["class"]), int(row["class_id"]))
        for row in rows
    )


def summarize(outputs, shape, cv2, np):
    check(isinstance(outputs, list) and outputs, "No Workflow output batch")
    result = []
    for output in outputs:
        rows = detection_list(output["predictions"])
        count = output["count"]
        check(
            count == len(rows),
            "PropertyDefinition count differs from serialized detections",
        )
        item = {
            "count": count,
            "classes": dict(Counter(row["class"] for row in rows)),
            "predictions": rows,
        }
        image, encoded = decode_visual(output["image"], cv2, np)
        plain, _ = decode_visual(output["plain"], cv2, np)
        check(
            image.shape == plain.shape == shape,
            "Visualization changed source dimensions",
        )
        changed = int(np.count_nonzero(np.any(image != plain, axis=2)))
        if rows:
            check(changed > 0, "Box/label visualization did not change any pixel")
        item["visualization"] = {
            **encoded,
            "width": image.shape[1],
            "height": image.shape[0],
            "changed_pixels_from_plain_encoding": changed,
        }
        for row in rows:
            check(
                0 <= row["confidence"] <= 1 and row["width"] > 0 and row["height"] > 0,
                "Invalid confidence or box dimensions",
            )
            check(
                0 <= row["x"] <= shape[1] and 0 <= row["y"] <= shape[0],
                "Detection center is outside original image coordinates",
            )
        result.append(item)
    return result


def source_manifest():
    import inference_sdk.http.client as client_module
    import inference_sdk.http.entities as entities_module
    import inference_sdk.http.utils.encoding as encoding_module
    from inference_sdk import InferenceHTTPClient

    files = {}
    for module in (client_module, entities_module, encoding_module):
        path = Path(inspect.getfile(module)).resolve()
        files[module.__name__] = {
            "path": str(path),
            "sha256": digest(path.read_bytes()),
        }
    return InferenceHTTPClient, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.66.80:9001")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="yolo8n/1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    origin = urlsplit(args.base_url)
    check(
        origin.scheme in ("http", "https")
        and origin.hostname
        and not origin.username
        and not origin.password
        and origin.path in ("", "/")
        and not origin.query
        and not origin.fragment,
        "Use a plain HTTP(S) API origin, never credentials in the URL",
    )
    os.environ["INFERENCE_RUNTIME_PROFILE"] = "full"
    import cv2
    import numpy as np

    from inference_sdk import InferenceConfiguration
    from inference_sdk.http.errors import HTTPCallErrorError

    client_type, sdk_files = source_manifest()
    image = cv2.imread(str(args.image))
    check(image is not None, "Cannot read the prepared input image")
    check(
        image.shape[0] * image.shape[1] * 4 <= 4096 * 2160,
        "Choose a source no larger than 1920x1080 for two-batch paired image outputs",
    )
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "api_origin": args.base_url,
        "model_id": args.model_id,
        "sdk_sources": sdk_files,
        "image": {
            "path": str(args.image.resolve()),
            "sha256": digest(args.image.read_bytes()),
            "width": image.shape[1],
            "height": image.shape[0],
        },
        "http": [],
        "cases": [],
        "test_workflow_ids": [],
        "cleanup": [],
        "camera_operations": False,
        "passed": False,
    }
    specs = {f"v{version}": graph(args.model_id, version) for version in (1, 2, 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".specifications.json").write_text(
        json.dumps(specs, indent=2) + "\n"
    )
    if args.prepare_only:
        report["preparation_only"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Prepared official-SDK HTTP suite; no device requests made")
        return 0
    token = args.token_file.read_text().strip()
    check(bool(token), "Device access password is empty")
    api = API(args.base_url.rstrip("/"), token, report)
    owned = []
    idle_claimed = False
    prefix = (
        "accept-sdk-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-")
        + uuid4().hex[:10]
    )

    def save(name, specification):
        identifier = prefix + "-" + name
        api.call("GET", "/build/api/" + identifier, expected=404)
        api.call(
            "POST",
            "/build/api/" + identifier,
            {
                "id": identifier,
                "name": "HTTP acceptance " + name,
                "config": json.dumps({"specification": specification}),
            },
            expected=201,
        )
        owned.append(identifier)
        report["test_workflow_ids"].append(identifier)
        saved = api.call("GET", "/build/api/" + identifier)
        config = saved["data"]["config"]
        original = json.loads(config["config"])["specification"]
        check(
            original == specification,
            "Original Builder config was not preserved on save/read",
        )
        return identifier

    def sdk_case(
        name, transport, *, spec=None, saved=None, images=None, parameters=None
    ):
        client = client_type(api_url=args.base_url.rstrip("/"), api_key=token)
        client.configure(
            InferenceConfiguration(
                api_key_transport=transport, workflow_run_retries_enabled=False
            )
        )
        event = {
            "name": name,
            "transport": transport,
            "mode": "inline" if spec is not None else "local",
            "passed": False,
        }
        report["cases"].append(event)
        started = time.monotonic()
        try:
            kwargs = (
                {"specification": spec}
                if spec is not None
                else {"workspace_name": "local", "workflow_id": saved}
            )
            with deadline(90):
                outputs = client.run_workflow(
                    **kwargs,
                    images={"image": image} if images is None else images,
                    parameters=parameters or {},
                )
            event["outputs"] = summarize(outputs, image.shape, cv2, np)
            event["passed"] = True
            return event["outputs"]
        finally:
            event["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)

    try:
        pipelines = api.call("GET", "/inference_pipelines/list")
        check(
            not pipelines.get("pipelines"),
            "A video pipeline is active; coordinator must reserve idle HTTP test time",
        )
        idle_claimed = True
        api.call("GET", "/healthz")
        api.call("GET", "/model/registry")
        baseline = None
        for version in (1, 2, 3):
            spec = specs[f"v{version}"]
            api.call("POST", "/workflows/validate", spec)
            saved = save(f"v{version}", spec)
            interface = api.call(
                "POST", "/local/workflows/" + saved + "/describe_interface", {}
            )
            check("outputs" in interface, "Named interface response is missing outputs")
            for transport in ("legacy", "header", "both"):
                for mode in ("inline", "local"):
                    result = sdk_case(
                        f"v{version}-{transport}-{mode}",
                        transport,
                        **({"spec": spec} if mode == "inline" else {"saved": saved}),
                    )
                    rows = result[0]["predictions"]
                    check(
                        rows,
                        "Real fixture produced zero detections; choose a useful acceptance image",
                    )
                    if baseline is None:
                        baseline = rows
                    check(
                        canonical(rows) == canonical(baseline),
                        "Standard detector versions/transports changed detections",
                    )
        chosen_class = baseline[0]["class"]
        expected = [row for row in baseline if row["class"] == chosen_class]
        result = sdk_case(
            "class-parameter-positive",
            "header",
            spec=specs["v3"],
            parameters={"classes": [chosen_class]},
        )
        check(
            canonical(result[0]["predictions"]) == canonical(expected),
            "Class selector parameter mismatch",
        )
        result = sdk_case(
            "class-parameter-empty-match",
            "header",
            spec=specs["v3"],
            parameters={"classes": ["__acceptance_no_such_class__"]},
        )
        check(result[0]["count"] == 0, "Absent class returned detections")
        high = min(1.0, max(row["confidence"] for row in baseline) + 0.0001)
        result = sdk_case(
            "confidence-parameter",
            "header",
            spec=specs["v3"],
            parameters={"confidence": high},
        )
        check(
            canonical(result[0]["predictions"])
            == canonical([row for row in baseline if row["confidence"] >= high]),
            "Confidence selector parameter mismatch",
        )
        filtered = filtered_graph(specs["v3"], chosen_class, 0.25)
        api.call("POST", "/workflows/validate", filtered)
        result = sdk_case("original-filter-node", "header", spec=filtered)
        check(
            canonical(result[0]["predictions"]) == canonical(expected),
            "Original class/confidence filter mismatch",
        )
        result = sdk_case(
            "two-images-one-workflow-batch",
            "header",
            spec=specs["v3"],
            images={"image": [image, image]},
        )
        check(
            len(result) == 2
            and all(
                canonical(item["predictions"]) == canonical(baseline) for item in result
            ),
            "Two-image batch does not match independent images",
        )
        nested = nested_graph(owned[2])
        nested_id = save("nested", nested)
        api.call("POST", "/workflows/validate", nested)
        result = sdk_case("saved-nested-workflow", "header", saved=nested_id)
        check(
            canonical(result[0]["predictions"]) == canonical(baseline),
            "Nested Workflow changed detections",
        )
        invalid = copy.deepcopy(specs["v3"])
        invalid["steps"][0]["images"] = "$steps.missing.image"
        error = api.call("POST", "/workflows/validate", invalid, expected=400)
        report["validation_error"] = {
            key: error.get(key) for key in ("error_type", "context", "blocks_errors")
        }
        check(error.get("blocks_errors"), "Original block-error context is missing")
        client = client_type(api_url=args.base_url.rstrip("/"), api_key=token)
        client.configure(
            InferenceConfiguration(
                api_key_transport="header", workflow_run_retries_enabled=False
            )
        )
        try:
            with deadline(90):
                client.run_workflow(specification=specs["v3"], images={})
        except HTTPCallErrorError as exc:
            report["sdk_missing_image_error"] = {
                "type": type(exc).__name__,
                "status": exc.status_code,
            }
            check(
                exc.status_code == 400,
                "Missing image did not preserve original input-validation status",
            )
        else:
            raise AssertionError("Missing image was accepted")
        check(
            not api.call("GET", "/inference_pipelines/list").get("pipelines"),
            "HTTP acceptance unexpectedly created a video pipeline",
        )
        report["passed"] = True
    except Exception as exc:
        # SDK exception text may contain HTTP response bodies; never dump it.
        report["failure"] = {
            "type": type(exc).__name__,
            "status": getattr(exc, "status_code", None),
        }
        if isinstance(exc, AssertionError):
            report["failure"]["check"] = str(exc)[:500]
    finally:
        if idle_claimed:
            try:
                api.call("POST", "/model/clear", {})
                report["model_released"] = True
            except Exception as exc:
                report["passed"] = False
                report["model_released"] = False
                report["model_release_error"] = type(exc).__name__
        for identifier in reversed(owned):
            try:
                api.call("DELETE", "/build/api/" + identifier)
                report["cleanup"].append({"workflow_id": identifier, "removed": True})
            except Exception as exc:
                report["passed"] = False
                report["cleanup"].append(
                    {
                        "workflow_id": identifier,
                        "removed": False,
                        "error_type": type(exc).__name__,
                    }
                )
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        # No password, image base64, SDK request body or private exception dump.
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "report": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
