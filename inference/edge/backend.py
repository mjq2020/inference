"""NumPy detection backend using only appmgr's remote RKNN service.

Kit imports happen at model load, so health and metadata inspection also work
on a host without Kit or an NPU. There is intentionally no local-runtime or
CPU model fallback.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from inference.edge.errors import EdgeError


def _load_kit():
    try:
        from kit.runtime.engine import ModelSpec, TensorSpec
        from kit.runtime.postprocess import detect
        from kit.runtime.preprocess import letterbox, preprocess
        from kit.runtime.remote import RemoteRknnSession
    except ImportError as exc:
        raise EdgeError(
            "The platform Kit runtime is unavailable.",
            code="kit_runtime_unavailable",
            status_code=503,
        ) from exc
    return SimpleNamespace(
        session=RemoteRknnSession,
        model_spec=ModelSpec,
        tensor_spec=TensorSpec,
        preprocess=preprocess,
        letterbox=letterbox,
        nms=detect.nms,
        decode_dfl=detect._decode_dfl,
    )


def _managed_endpoint():
    endpoint = (
        os.environ.get("RECAMERA_INFERENCE_SERVICE_SOCK")
        or os.environ.get("RECAMERA_INFERENCE_SERVICE")
        or ""
    ).strip()
    generation = os.environ.get("RECAMERA_APP_GENERATION", "")
    if (
        not endpoint
        or not os.environ.get("RECAMERA_APP_ID")
        or not os.environ.get("RECAMERA_APP_INSTANCE")
        or not generation.isascii()
        or not generation.isdecimal()
        or int(generation) <= 0
    ):
        raise EdgeError(
            "NPU model loading requires an appmgr-authorized application launch; "
            "the daemon must authorize this model path and SHA-256.",
            code="npu_authorization_required",
            status_code=403,
        )
    return endpoint


def _runtime_error(exc, operation):
    if isinstance(exc, EdgeError):
        return exc
    code = getattr(exc, "code", "")
    if isinstance(exc, PermissionError) or code in {
        "unauthorized",
        "authorization_pending",
        "missing_managed_generation",
    }:
        return EdgeError(
            str(exc),
            code="npu_authorization_required",
            status_code=403,
            details={"operation": operation, "platform_code": code},
        )
    if code in {"queue_full", "resource_busy", "memory_budget_exceeded"}:
        status = 429
    elif isinstance(exc, (OSError, ConnectionError)) or "unavailable" in code:
        status = 503
    elif code in {"model_digest_mismatch", "model_not_found", "invalid_input"}:
        status = 400
    else:
        status = 502
    return EdgeError(
        f"Platform RKNN {operation} failed: {exc}",
        code=code or f"rknn_{operation}_failed",
        status_code=status,
        details={"operation": operation},
    )


class KitBackend:
    def __init__(self, metadata):
        self.metadata = metadata
        self._session = None
        self._released = False
        self._release_failed = False
        endpoint = _managed_endpoint()
        self._kit = _load_kit()

        def tensor(spec, *, name=None):
            return self._kit.tensor_spec(
                name=name or spec.name,
                shape=spec.shape,
                dtype=spec.dtype,
                layout=spec.layout,
            )

        try:
            spec = self._kit.model_spec(
                path=str(metadata.path),
                name=metadata.model_id,
                # appmgr's trusted contract uses the generic positional name
                # "input"; the graph's internal input name stays in metadata.
                inputs=(tensor(metadata.input, name="input"),),
                outputs=tuple(tensor(item) for item in metadata.outputs),
            )
            self._session = self._kit.session(
                spec,
                socket_path=endpoint,
                model_sha256=metadata.sha256,
                verify_model=True,
                strict_inputs=True,
                memory_mb=metadata.memory_mb,
            )
        except Exception as exc:
            # RemoteRknnSession closes its transport on constructor failure.
            raise _runtime_error(exc, "load") from exc

    def infer(
        self,
        image,
        *,
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_filter=None,
        class_agnostic_nms=False,
        max_candidates=3000,
        **kwargs,
    ):
        if self._released:
            raise EdgeError(
                "RKNN session has been released.",
                code="session_released",
                status_code=409,
            )
        if self._release_failed:
            raise EdgeError(
                "Retry unloading the model after its failed release before inference.",
                code="session_release_pending",
                status_code=409,
            )
        if kwargs:
            raise EdgeError(
                "Unsupported inference options.",
                code="unsupported_option",
                details={"options": sorted(kwargs)},
            )
        for name, value in (
            ("confidence", confidence),
            ("iou_threshold", iou_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise EdgeError(
                    f"{name} must be a finite number in 0..1.", code="invalid_input"
                )
        for name, value, limit in (
            ("max_detections", max_detections, 1000),
            ("max_candidates", max_candidates, 10000),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise EdgeError(
                    f"{name} must be an integer in 1..{limit}.", code="invalid_input"
                )
        if not isinstance(class_agnostic_nms, bool):
            raise EdgeError("class_agnostic_nms must be boolean.", code="invalid_input")
        if class_filter is not None and (
            not isinstance(class_filter, (list, tuple))
            or any(not isinstance(v, str) for v in class_filter)
        ):
            raise EdgeError(
                "class_filter must be a list of class names.", code="invalid_input"
            )
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[0] * image.shape[1] > 16_000_000
        ):
            raise EdgeError(
                "Expected a non-empty BGR HWC uint8 image (up to 16 megapixels).",
                code="invalid_input",
            )
        started = time.monotonic()
        try:
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            if self.metadata.padding_value == 114:
                inputs, info = self._kit.preprocess(
                    rgb, new_shape=self.metadata.input_size
                )
            else:
                padded, info = self._kit.letterbox(
                    rgb,
                    new_shape=self.metadata.input_size,
                    color=self.metadata.padding_value,
                )
                inputs = np.expand_dims(padded, 0)
            if inputs.dtype != np.uint8 or inputs.shape != self.metadata.input.shape:
                raise EdgeError(
                    "Preprocessing violated the RKNN input contract.",
                    code="invalid_preprocess_output",
                    status_code=502,
                )
            outputs = self._session.infer(inputs)
        except Exception as exc:
            raise _runtime_error(exc, "infer") from exc
        try:
            self._validate_outputs(outputs)
            boxes, scores, class_ids = self._decode(outputs, confidence)
            if class_filter is not None:
                allowed = [
                    index
                    for index, label in enumerate(self.metadata.labels)
                    if label in class_filter
                ]
                selected = np.isin(class_ids, allowed)
                boxes, scores, class_ids = (
                    boxes[selected],
                    scores[selected],
                    class_ids[selected],
                )
            if len(scores) > max_candidates:
                selected = np.argpartition(scores, -max_candidates)[-max_candidates:]
                boxes, scores, class_ids = (
                    boxes[selected],
                    scores[selected],
                    class_ids[selected],
                )
            # End-to-end / fused-NMS exports already select their detections.
            # A second NMS changes the model's predictions (notably YOLO26).
            keep = list(range(len(scores))) if not self.metadata.apply_nms else []
            if self.metadata.apply_nms:
                groups = np.zeros_like(class_ids) if class_agnostic_nms else class_ids
                for group in np.unique(groups):
                    indices = np.flatnonzero(groups == group)
                    local_keep = self._kit.nms(
                        boxes[indices], scores[indices], iou_threshold
                    )
                    keep.extend(indices[local_keep].tolist())
            keep.sort(key=lambda index: float(scores[index]), reverse=True)
            predictions = []
            for index in keep[:max_detections]:
                box = boxes[index].astype(np.float64, copy=True)
                box[[0, 2]] = np.clip(
                    (box[[0, 2]] - info.pad_w) / info.scale, 0, image.shape[1]
                )
                box[[1, 3]] = np.clip(
                    (box[[1, 3]] - info.pad_h) / info.scale, 0, image.shape[0]
                )
                x1, y1, x2, y2 = box.tolist()
                cid = int(class_ids[index])
                predictions.append(
                    {
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "confidence": float(scores[index]),
                        "class": self.metadata.labels[cid],
                        "class_id": cid,
                        "detection_id": str(uuid4()),
                    }
                )
            return {
                "inference_id": str(uuid4()),
                "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
                "predictions": predictions,
                "time": time.monotonic() - started,
            }
        except Exception as exc:
            raise _runtime_error(exc, "postprocess") from exc

    def _validate_outputs(self, outputs):
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
            self.metadata.outputs
        ):
            raise EdgeError(
                "RKNN output count differs from model metadata.",
                code="invalid_model_output",
                status_code=502,
            )
        for spec, output in zip(self.metadata.outputs, outputs):
            if (
                not isinstance(output, np.ndarray)
                or output.dtype != np.float32
                or output.shape != spec.shape
                or not np.isfinite(output).all()
            ):
                raise EdgeError(
                    "RKNN output violates its shape/dtype/finite contract.",
                    code="invalid_model_output",
                    status_code=502,
                    details={"tensor": spec.name},
                )
            if self.metadata.decoder == "yolo-decoded":
                values = output[0] if spec.layout == "BCN" else output[0].T
                scores = values[4:]
                if (values[2:4] < 0).any():
                    raise EdgeError(
                        "Decoded box dimensions must be non-negative.",
                        code="invalid_model_output",
                        status_code=502,
                    )
            elif self.metadata.decoder == "yolo-end2end":
                values = output[0]
                scores, classes = values[:, 4], values[:, 5]
                if (
                    (values[:, 2:4] < values[:, :2]).any()
                    or (classes != np.floor(classes)).any()
                    or (classes < 0).any()
                    or (classes >= len(self.metadata.labels)).any()
                ):
                    raise EdgeError(
                        "End-to-end detections contain invalid xyxy boxes or class IDs.",
                        code="invalid_model_output",
                        status_code=502,
                    )
            elif self.metadata.decoder == "yolo-distance":
                if spec.role != "scores":
                    continue
                scores = output
            elif spec.role != "score_sum" and output.shape[1] == len(
                self.metadata.labels
            ):
                scores = output
            else:
                continue
            if self.metadata.score_format == "probabilities" and (
                (scores < 0).any() or (scores > 1).any()
            ):
                raise EdgeError(
                    "Declared probability output contains values outside 0..1.",
                    code="invalid_model_output",
                    status_code=502,
                )

    def _decode(self, outputs, confidence):
        if self.metadata.decoder == "yolo-end2end":
            values = outputs[0][0]
            selected = values[:, 4] >= confidence
            return (
                values[selected, :4],
                values[selected, 4],
                values[selected, 5].astype(np.int64),
            )
        if not self.metadata.apply_nms and self.metadata.decoder in {
            "yolo-distance",
            "yolo-dfl",
        }:
            return self._decode_end2end_heads(outputs, confidence)
        if self.metadata.decoder == "yolo-distance":
            branches = {}
            for spec, output in zip(self.metadata.outputs, outputs):
                branches.setdefault(output.shape[2], {})[spec.role] = output[0]
            boxes, scores, classes = [], [], []
            for grid_size, branch in sorted(branches.items(), reverse=True):
                values = branch["scores"].reshape(len(self.metadata.labels), -1)
                class_ids = values.argmax(axis=0)
                best = values[class_ids, np.arange(values.shape[1])]
                if self.metadata.score_format == "logits":
                    best = np.exp(-np.logaddexp(np.float32(0), -best))
                keep = np.flatnonzero(best >= confidence)
                stride = self.metadata.input_size / grid_size
                centers = np.stack((keep % grid_size, keep // grid_size), axis=1) + 0.5
                distances = branch["boxes"].reshape(4, -1)[:, keep].T
                boxes.append(
                    np.concatenate(
                        (centers - distances[:, :2], centers + distances[:, 2:]), axis=1
                    )
                    * stride
                )
                scores.append(best[keep])
                classes.append(class_ids[keep])
            return (
                np.concatenate(boxes),
                np.concatenate(scores),
                np.concatenate(classes),
            )
        if self.metadata.decoder == "yolo-dfl":
            prepared = []
            for spec, output in zip(self.metadata.outputs, outputs):
                if spec.role == "score_sum":
                    # This is an explicitly declared optional export helper,
                    # never an inferred/silently discarded output tensor.
                    continue
                if self.metadata.score_format == "logits" and output.shape[1] == len(
                    self.metadata.labels
                ):
                    # The Kit helper guesses sigmoid from observed ranges;
                    # normalize explicitly so logits wholly inside [0,1]
                    # still receive sigmoid. Do not mutate borrowed outputs.
                    output = np.exp(-np.logaddexp(np.float32(0), -output))
                prepared.append(output)
            return self._kit.decode_dfl(
                prepared,
                confidence,
                input_size=self.metadata.input_size,
                reg_max=16,
                nc=len(self.metadata.labels),
            )
        spec = self.metadata.outputs[0]
        values = outputs[0][0].T if spec.layout == "BCN" else outputs[0][0]
        class_start = 5 if self.metadata.objectness else 4
        class_ids = values[:, class_start:].argmax(axis=1)
        scores = values[np.arange(values.shape[0]), class_ids + class_start]
        if self.metadata.score_format == "logits":
            scores = np.exp(-np.logaddexp(np.float32(0), -scores))
        if self.metadata.objectness:
            objectness = values[:, 4]
            if self.metadata.score_format == "logits":
                objectness = np.exp(-np.logaddexp(np.float32(0), -objectness))
            scores = scores * objectness
        selected = scores >= confidence
        xywh = values[selected, :4]
        half = xywh[:, 2:4] / 2
        boxes = np.concatenate((xywh[:, :2] - half, xywh[:, :2] + half), axis=1)
        return boxes, scores[selected], class_ids[selected]

    def _decode_end2end_heads(self, outputs, confidence):
        """Reproduce the export's anchor TopK then class TopK after head cutting.

        Selecting only argmax(class) loses secondary classes emitted by the
        source graph. Select before Workflow class filtering, just as the full
        end-to-end graph does. Only TopK anchors' class vectors are retained.
        """
        branches = {}
        for spec, output in zip(self.metadata.outputs, outputs):
            if spec.role == "score_sum":
                continue
            role = spec.role or ("boxes" if output.shape[1] == 64 else "scores")
            branches.setdefault(output.shape[2], {})[role] = output[0]
        boxes, scores = [], []
        limit = self.metadata.topk
        for grid_size, branch in sorted(branches.items(), reverse=True):
            values = branch["scores"].reshape(len(self.metadata.labels), -1).T
            best = values.max(axis=1)
            count = min(limit, len(best))
            selected = np.argpartition(best, -count)[-count:]
            values = values[selected]
            if self.metadata.score_format == "logits":
                values = np.exp(-np.logaddexp(np.float32(0), -values))
            distances = branch["boxes"].reshape(-1, grid_size * grid_size)[:, selected]
            if self.metadata.decoder == "yolo-dfl":
                distribution = distances.reshape(4, 16, count)
                weights = np.exp(distribution - distribution.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
                distances = (
                    weights * np.arange(16, dtype=np.float32)[None, :, None]
                ).sum(axis=1)
            centers = (
                np.stack((selected % grid_size, selected // grid_size), axis=1) + 0.5
            )
            boxes.append(
                np.concatenate(
                    (centers - distances[:2].T, centers + distances[2:].T), axis=1
                )
                * (self.metadata.input_size / grid_size)
            )
            scores.append(values)
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        count = min(limit, len(boxes))
        anchors = np.argpartition(scores.max(axis=1), -count)[-count:]
        boxes, scores = boxes[anchors], scores[anchors]
        flattened = scores.reshape(-1)
        count = min(limit, len(flattened))
        selected = np.argpartition(flattened, -count)[-count:]
        selected = selected[flattened[selected] >= confidence]
        return (
            boxes[selected // len(self.metadata.labels)],
            flattened[selected],
            selected % len(self.metadata.labels),
        )

    def release(self):
        if self._released:
            return
        try:
            self._session.release()
        except Exception as exc:
            self._release_failed = True
            raise _runtime_error(exc, "release") from exc
        self._released = True
        self._session = None

    close = release
