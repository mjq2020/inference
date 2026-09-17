#!/usr/bin/env python3
"""Validate an already running, appmgr-authorized application over localhost HTTP.

This script never installs applications, requests SDK grants, or imports RKNN.
With --image, OpenCV is used only to prepare a lossless ROI input for comparison.
Run from the application's Python environment when image preparation is needed:

    python test_device_runtime.py --model-id detector/1 --image /tmp/test.jpg \
        --require-detections --output /tmp/inference-device-evidence.json

Without --image, image comparisons are explicitly skipped; both camera modes
still run three complete start/pause/resume/stop cycles by default. An existing
active pipeline is a preflight failure and is never stopped by this script.
The JSON report contains detections and timings, but no image payload or token.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class ValidationFailure(RuntimeError):
    pass


class API:
    def __init__(self, args, report):
        self.base_url = args.base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username
            or parsed.password
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("--base-url must be a localhost HTTP origin")
        self.timeout = args.request_timeout
        self.token = os.getenv(args.token_env, "")
        self.report = report

        # A localhost test must not follow proxy configuration or redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def call(self, path, body=None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            if len(data) > 8 * 1024 * 1024:
                raise ValidationFailure("Test request exceeds the default 8 MiB limit")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers
        )
        event = {"method": request.get_method(), "path": path}
        self.report["requests"].append(event)
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                event["status"] = response.status
                payload = response.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise ValidationFailure("API response exceeds test-client byte budget")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            event["status"] = exc.code
            event["response"] = exc.read(8192).decode(errors="replace")
            raise ValidationFailure(
                f"{request.get_method()} {path}: HTTP {exc.code}: {event['response']}"
            ) from exc
        finally:
            event["duration_ms"] = round((time.monotonic() - started) * 1000, 2)


def image_value(data):
    return {"type": "base64", "value": base64.b64encode(data).decode("ascii")}


def predictions(response):
    result = response.get("predictions")
    if not isinstance(result, list):
        raise ValidationFailure("Expected an object-detection predictions list")
    return result


def workflow(model_id, parameters, roi=None):
    steps = []
    if roi:
        steps.append(
            {
                "type": "AbsoluteStaticCrop",
                "name": "roi",
                "image": "$inputs.image",
                **roi,
            }
        )
    steps.extend(
        [
            {
                "type": "ObjectDetectionModel",
                "name": "detect",
                "images": "$steps.roi.crops" if roi else "$inputs.image",
                "model_id": model_id,
                **parameters,
            },
            {
                "type": "PropertyDefinition",
                "name": "count",
                "data": "$steps.detect.predictions",
                "operations": [{"type": "SequenceLength"}],
            },
        ]
    )
    outputs = [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detect.predictions",
        },
        {"type": "JsonField", "name": "count", "selector": "$steps.count.output"},
    ]
    if roi:
        outputs.append(
            {
                "type": "JsonField",
                "name": "own",
                "selector": "$steps.detect.predictions",
                "coordinates_system": "own",
            }
        )
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": outputs,
    }


class DeviceValidation:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.api = API(args, report)
        self.pipeline_id = None
        self.owns_model = False
        self.parameters = {
            "confidence": args.confidence,
            "iou_threshold": 0.45,
            "max_detections": 32,
            "class_agnostic_nms": False,
        }

    def check(self, condition, name, **details):
        self.report["checks"].append(
            {"name": name, "passed": bool(condition), **details}
        )
        if not condition:
            raise ValidationFailure(name)

    def registry(self):
        return self.api.call("/model/registry")["models"]

    def state(self):
        states = self.api.call("/inference_pipelines/list")["pipelines"]
        return next(
            (item for item in states if item["pipeline_id"] == self.pipeline_id), {}
        )

    def command(self, name):
        return self.api.call(
            "/inference_pipelines/" + name, {"pipeline_id": self.pipeline_id}
        )

    def wait_frames(self, target):
        deadline = time.monotonic() + self.args.frame_timeout
        while time.monotonic() < deadline:
            state = self.state()
            if state.get("status") == "failed":
                raise ValidationFailure(f"Camera pipeline failed: {state}")
            if state.get("frames", 0) >= target:
                return state
            if state.get("status") not in ("running", "paused"):
                raise ValidationFailure(f"Camera stopped before frame target: {state}")
            time.sleep(0.15)
        raise ValidationFailure(f"Timed out waiting for {target} camera frames")

    def stable_pause(self):
        # A frame already in inference may finish after pause returns. Require
        # a subsequent stable interval, rather than racing that in-flight frame.
        deadline = time.monotonic() + self.args.frame_timeout
        state = self.state()
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.15)
            current = self.state()
            self.check(current.get("status") == "paused", "camera_remains_paused")
            if current["frames"] != state["frames"]:
                stable_since = time.monotonic()
                state = current
            if time.monotonic() - stable_since >= self.args.pause_seconds:
                return current
        raise ValidationFailure("Paused camera did not reach a stable frame count")

    def compare(self, expected, actual, name, offset=(0, 0)):
        left, right = predictions(expected), predictions(actual)
        self.check(len(left) == len(right), name + "_count", count=len(left))
        # RKNN/NMS ordering is stable for identical pixels and options. Ignore
        # timing, inference IDs and random detection IDs, retaining numeric data.
        for index, (a, b) in enumerate(zip(left, right)):
            self.check(
                a["class_id"] == b["class_id"] and a["class"] == b["class"],
                name + "_class",
                index=index,
            )
            for key in ("x", "y", "width", "height", "confidence"):
                shift = offset[0] if key == "x" else offset[1] if key == "y" else 0
                tolerance = 1e-5 if key == "confidence" else 0.001
                self.check(
                    math.isclose(
                        float(a[key]) + shift,
                        float(b[key]),
                        abs_tol=tolerance,
                        rel_tol=1e-6,
                    ),
                    name + "_" + key,
                    index=index,
                )

    def run_workflow(self, specification, value):
        result = self.api.call(
            "/workflows/run",
            {"specification": specification, "inputs": {"image": value}},
        )
        self.check(len(result["outputs"]) == 1, "workflow_single_input_single_output")
        output = result["outputs"][0]
        self.check(
            output["count"] == len(predictions(output["predictions"])),
            "workflow_count_matches_predictions",
            count=output["count"],
        )
        return output

    def test_image(self):
        import cv2
        import numpy as np

        source = Path(self.args.image)
        raw = source.read_bytes()
        if len(raw) > 6 * 1024 * 1024:
            raise ValidationFailure("Choose an image below 6 MiB for base64 transport")
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValidationFailure("OpenCV could not decode the test image")
        height, width = decoded.shape[:2]
        self.check(min(height, width) >= 4, "image_large_enough_for_roi")
        evidence = {
            "status": "running",
            "path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": width,
            "height": height,
        }
        self.report["image"] = evidence
        self.owns_model = True
        direct = self.api.call(
            "/infer/object_detection",
            {"model_id": self.model_id, "image": image_value(raw), **self.parameters},
        )
        evidence["direct"] = direct
        full = self.run_workflow(self.specification, image_value(raw))
        evidence["workflow"] = full
        self.compare(direct, full["predictions"], "direct_vs_workflow")
        if self.args.require_detections:
            self.check(bool(predictions(direct)), "real_image_has_detections")
        crop_width, crop_height = 2 * (width // 4), 2 * (height // 4)
        roi = {
            "x_center": width // 2,
            "y_center": height // 2,
            "width": crop_width,
            "height": crop_height,
        }
        x, y = roi["x_center"] - crop_width // 2, roi["y_center"] - crop_height // 2
        ok, encoded = cv2.imencode(
            ".png", decoded[y : y + crop_height, x : x + crop_width]
        )
        self.check(ok, "roi_png_encoded")
        roi_direct = self.api.call(
            "/infer/object_detection",
            {
                "model_id": self.model_id,
                "image": image_value(encoded.tobytes()),
                **self.parameters,
            },
        )
        roi_workflow = self.run_workflow(
            workflow(self.model_id, self.parameters, roi=roi), image_value(raw)
        )
        evidence["roi"] = {
            "rectangle": {"left": x, "top": y, **roi},
            "direct": roi_direct,
            "workflow": roi_workflow,
            "coordinate_check": (
                "nonempty_predictions"
                if predictions(roi_direct)
                else "empty_predictions"
            ),
        }
        self.compare(roi_direct, roi_workflow["own"], "roi_direct_vs_workflow_own")
        self.compare(
            roi_workflow["own"],
            roi_workflow["predictions"],
            "roi_own_vs_root",
            offset=(x, y),
        )
        evidence["status"] = "passed"

    def stop_owned_pipeline(self):
        if not self.pipeline_id:
            return None
        state = self.command("terminate")
        self.check(state["status"] == "stopped", "camera_stopped", state=state)
        self.check(state["retained_results"] == 0, "stop_discards_retained_results")
        models = self.registry()
        self.check(not any(item["loaded"] for item in models), "stop_released_models")
        self.pipeline_id = None
        self.owns_model = False
        return {"state": state, "models": models}

    def test_camera_cycle(self, mode, cycle):
        evidence = {"mode": mode, "cycle": cycle, "status": "running"}
        self.report["camera"].append(evidence)
        request = {"max_fps": self.args.fps}
        request.update(
            {"model_id": self.model_id}
            if mode == "model"
            else {"specification": self.specification, "image_input": "image"}
        )
        previous_ids = {
            item["pipeline_id"]
            for item in self.api.call("/inference_pipelines/list")["pipelines"]
        }
        self.owns_model = True
        try:
            try:
                started = self.api.call("/inference_pipelines/initialise", request)
                self.pipeline_id = started["pipeline_id"]
                evidence["start"] = started
            except Exception:
                # If a response was lost after initialization, identify only a
                # newly created pipeline from this dedicated test's request.
                states = self.api.call("/inference_pipelines/list")["pipelines"]
                new = [
                    item for item in states if item["pipeline_id"] not in previous_ids
                ]
                if len(new) == 1:
                    self.pipeline_id = new[0]["pipeline_id"]
                raise
            evidence["before_pause"] = self.wait_frames(4)
            evidence["pause"] = self.command("pause")
            paused = self.stable_pause()
            evidence["stable_pause"] = paused
            self.check(
                paused["retained_results"] == 2, "camera_retains_only_two_results"
            )
            results = self.command("consume")["results"]
            evidence["results"] = results
            self.check(len(results) == 2, "consume_returns_two_results")
            self.check(
                [item["frame_id"] for item in results]
                == [paused["frames"] - 2, paused["frames"] - 1],
                "consume_returns_latest_frames",
            )
            self.check(
                self.command("consume")["results"] == [], "consume_drains_result_queue"
            )
            for record in results:
                self.check(
                    record["pipeline_id"] == self.pipeline_id, "result_pipeline_id"
                )
                if mode == "workflow":
                    self.check(len(record["result"]) == 1, "camera_workflow_batch_one")
                    output = record["result"][0]
                    self.check(
                        output["count"] == len(predictions(output["predictions"])),
                        "camera_workflow_count_matches_predictions",
                    )
                else:
                    predictions(record["result"])
            evidence["resume"] = self.command("resume")
            evidence["after_resume"] = self.wait_frames(paused["frames"] + 2)
            evidence["stop"] = self.stop_owned_pipeline()
            evidence["status"] = "passed"
        finally:
            if self.pipeline_id:
                evidence["cleanup"] = self.stop_owned_pipeline()

    def run(self):
        self.report["health"] = self.api.call("/healthz")
        self.check(
            self.report["health"].get("profile") == "rv1126b"
            and self.report["health"].get("backend") == "rknn",
            "rknn_device_profile",
        )
        self.report["capabilities"] = self.api.call("/capabilities")
        existing = self.api.call("/inference_pipelines/list")["pipelines"]
        self.check(
            not any(item["status"] in ("running", "paused") for item in existing),
            "no_preexisting_active_pipeline",
        )
        models = self.registry()
        self.report["initial_models"] = models
        self.model_id = self.args.model_id
        if not self.model_id:
            self.check(len(models) == 1, "use_model_id_when_registry_is_not_unique")
            self.model_id = models[0]["model_id"]
        self.check(
            any(item["model_id"] == self.model_id for item in models),
            "requested_model_is_installed",
        )
        self.report["model_id"] = self.model_id
        self.specification = workflow(self.model_id, self.parameters)
        self.report["workflow_specification"] = self.specification
        if self.args.image:
            self.test_image()
        else:
            self.report["image"] = {
                "status": "skipped",
                "reason": "--image not supplied",
            }
        if self.args.skip_camera:
            self.report["camera_skipped"] = "--skip-camera supplied"
        else:
            for mode in ("model", "workflow"):
                for cycle in range(1, self.args.cycles + 1):
                    self.test_camera_cycle(mode, cycle)

    def cleanup(self):
        if self.pipeline_id:
            self.report["final_pipeline_cleanup"] = self.stop_owned_pipeline()
        if self.owns_model:
            self.api.call("/model/clear", {})
            models = self.registry()
            self.check(
                not any(item["loaded"] for item in models), "final_models_released"
            )
            self.report["final_models"] = models
            self.owns_model = False


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9001")
    parser.add_argument(
        "--model-id", help="Required if more than one model is installed"
    )
    parser.add_argument(
        "--image", help="Real PNG/JPEG file; omitted means lifecycle only"
    )
    parser.add_argument("--require-detections", action="store_true")
    parser.add_argument(
        "--skip-camera", action="store_true", help="Run image checks only"
    )
    parser.add_argument("--cycles", type=int, default=3, help="Cycles per camera mode")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--token-env", default="INFERENCE_EDGE_API_TOKEN")
    parser.add_argument(
        "--output", help="Also save the complete JSON report to this file"
    )
    args = parser.parse_args()
    if (
        args.cycles < 1
        or min(args.fps, args.frame_timeout, args.request_timeout, args.pause_seconds)
        <= 0
    ):
        parser.error("cycles, fps and timeout/interval values must be positive")
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    if not args.image and (args.require_detections or args.skip_camera):
        parser.error("--require-detections and --skip-camera require --image")
    return args


def main():
    args = arguments()
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "base_url": args.base_url,
        "cycles_per_mode": args.cycles,
        "checks": [],
        "requests": [],
        "camera": [],
    }
    validation = None
    try:
        validation = DeviceValidation(args, report)
        validation.run()
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if validation:
            try:
                validation.cleanup()
            except Exception as exc:
                report["status"] = "failed"
                report["cleanup_error"] = str(exc)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
