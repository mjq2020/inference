"""Read-only RKNN package metadata and a single-session model manager.

Packages live at ``<root>/<model_id>/model.json``. Metadata is descriptive,
never an appmgr permission grant. The inference daemon additionally verifies
the installed application's authorized model path and digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from inference.edge.errors import EdgeError


def _invalid(message):
    return EdgeError(message, code="invalid_model_metadata")


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    role: str | None = None

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise _invalid("Tensor metadata must be an object.")
        name, shape = value.get("name"), value.get("shape")
        if not isinstance(name, str) or not name.strip():
            raise _invalid("Each tensor needs a non-empty name.")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(v) is not int or v <= 0 for v in shape)
        ):
            raise _invalid("Tensor shapes must contain fixed positive integers.")
        dtype, layout = value.get("dtype"), value.get("layout")
        if dtype not in {"uint8", "float32"} or not isinstance(layout, str):
            raise _invalid("Unsupported tensor dtype or layout.")
        role = value.get("role")
        if role not in {None, "boxes", "scores", "score_sum", "detections"}:
            raise _invalid("Unsupported tensor role.")
        return cls(name, tuple(shape), dtype, layout, role)


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    path: Path
    sha256: str
    labels: tuple[str, ...]
    input: TensorMetadata
    outputs: tuple[TensorMetadata, ...]
    decoder: str
    score_format: str = "probabilities"
    memory_mb: int = 64
    padding_value: int = 114
    objectness: bool = False
    apply_nms: bool = True
    topk: int = 300

    @property
    def input_size(self):
        return self.input.shape[1]

    def describe(self):
        return {
            "model_id": self.model_id,
            "task_type": "object-detection",
            "platform": "rv1126b",
            "format": "rknn",
            "sha256": self.sha256,
            "labels": list(self.labels),
            "input_shape": list(self.input.shape),
            "input_dtype": self.input.dtype,
            "input_layout": self.input.layout,
            "decoder": self.decoder,
            "score_format": self.score_format,
            "objectness": self.objectness,
        }


class LocalModelStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def _model_dir(self, model_id):
        if (
            not isinstance(model_id, str)
            or len(model_id) > 160
            or not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*", model_id)
            or any(p in {".", ".."} for p in model_id.split("/"))
        ):
            raise EdgeError("Invalid local model ID.", code="invalid_model_id")
        path = (self.root / model_id).resolve()
        if not path.is_relative_to(self.root):
            raise _invalid("Model directory must remain inside the model store.")
        return path

    def get(self, model_id: str, *, verify=True) -> ModelMetadata:
        directory = self._model_dir(model_id)
        metadata_file = (directory / "model.json").resolve()
        if not metadata_file.is_relative_to(directory):
            raise _invalid("Metadata must remain inside its model package.")
        try:
            if metadata_file.stat().st_size > 1024 * 1024:
                raise _invalid("Model metadata exceeds 1 MiB.")
            value = json.loads(metadata_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EdgeError(
                f"Local RKNN model {model_id!r} was not found.",
                code="model_not_found",
                status_code=404,
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _invalid("Cannot read valid model metadata.") from exc
        metadata = self._parse(model_id, directory, value)
        if verify:
            try:
                digest = hashlib.sha256()
                with metadata.path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != metadata.sha256:
                    raise EdgeError(
                        "RKNN file SHA-256 does not match its metadata.",
                        code="model_digest_mismatch",
                    )
            except OSError as exc:
                raise EdgeError(
                    "Cannot read RKNN model asset.",
                    code="model_not_found",
                    status_code=404,
                ) from exc
        return metadata

    def _parse(self, model_id, directory, value):
        if not isinstance(value, dict):
            raise _invalid("Model metadata must be an object.")
        required = {
            "schema_version": 1,
            "model_id": model_id,
            "platform": "rv1126b",
            "format": "rknn",
            "task": "object-detection",
        }
        if type(value.get("schema_version")) is not int or any(
            value.get(k) != v for k, v in required.items()
        ):
            raise _invalid(
                "Require schema_version 1, matching model_id, rv1126b RKNN object-detection metadata."
            )
        asset = value.get("model_file")
        if not isinstance(asset, str) or Path(asset).is_absolute():
            raise _invalid("model_file must be relative to its model package.")
        path = (directory / asset).resolve()
        if not path.is_relative_to(directory) or path.suffix != ".rknn":
            raise _invalid("Model asset must be a .rknn file inside its package.")
        sha = value.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise _invalid("sha256 must be 64 lowercase hexadecimal characters.")
        labels = value.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or len(labels) > 1000
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise _invalid("labels must be a non-empty list of class names.")
        raw_input = value.get("input")
        input_spec = TensorMetadata.parse(raw_input)
        shape = input_spec.shape
        padding = raw_input.get("padding_value", 114)
        if type(padding) is not int or not 0 <= padding <= 255:
            raise _invalid("Letterbox padding must be an integer in 0..255.")
        if (
            input_spec.dtype != "uint8"
            or input_spec.layout != "NHWC"
            or len(shape) != 4
            or shape[0] != 1
            or shape[3] != 3
            or shape[1] != shape[2]
            or shape[1] > 1280
            or raw_input.get("color_format") != "RGB"
            or raw_input.get("normalization") != "baked"
        ):
            raise _invalid(
                "Require batch-1 square NHWC RGB uint8 input (<=1280), with normalization baked into RKNN."
            )
        raw_outputs = value.get("outputs")
        if (
            not isinstance(raw_outputs, list)
            or not raw_outputs
            or len(raw_outputs) > 12
        ):
            raise _invalid("Declare the model's output tensors.")
        outputs = tuple(TensorMetadata.parse(v) for v in raw_outputs)
        if len({o.name for o in outputs}) != len(outputs) or any(
            o.dtype != "float32" for o in outputs
        ):
            raise _invalid("Outputs need unique names and float32 runtime values.")
        if sum(math.prod(o.shape) * 4 for o in outputs) > 64 * 1024 * 1024:
            raise _invalid("Declared output tensors exceed the 64 MiB edge limit.")
        post = value.get("postprocess")
        if not isinstance(post, dict) or post.get("scores") not in {
            "probabilities",
            "logits",
        }:
            raise _invalid(
                "Declare postprocess scores explicitly as probabilities or logits."
            )
        decoder = post.get("kind")
        nc = len(labels)
        apply_nms = post.get("nms", decoder != "yolo-end2end")
        if type(apply_nms) is not bool or (decoder == "yolo-end2end" and apply_nms):
            raise _invalid(
                "NMS must be a boolean and must be disabled for end-to-end outputs."
            )
        topk = post.get("topk", 300)
        if type(topk) is not int or not 1 <= topk <= 1000:
            raise _invalid("End-to-end topk must be an integer in 1..1000.")
        if "topk" in post and (
            apply_nms or decoder not in {"yolo-distance", "yolo-dfl"}
        ):
            raise _invalid("topk only applies to raw end-to-end detection heads.")
        if decoder == "yolo-decoded":
            if post.get("box_format") != "xywh" or len(outputs) != 1:
                raise _invalid("yolo-decoded requires one pixel xywh output.")
            out = outputs[0]
            if out.role not in {None, "detections"}:
                raise _invalid("Decoded output role must be detections.")
            expected_axis = 1 if out.layout == "BCN" else 2
            objectness = post.get("objectness", False)
            channels = nc + 4 + int(objectness is True)
            if (
                type(objectness) is not bool
                or out.layout not in {"BCN", "BNC"}
                or len(out.shape) != 3
                or out.shape[0] != 1
                or out.shape[expected_axis] != channels
                or out.shape[3 - expected_axis] <= channels
            ):
                raise _invalid(
                    "Decoded output must match its BCN/BNC box, optional objectness and class channels."
                )
        elif decoder == "yolo-end2end":
            out = outputs[0]
            if (
                len(outputs) != 1
                or post.get("box_format") != "xyxy"
                or post.get("scores") != "probabilities"
                or out.layout != "BNC"
                or len(out.shape) != 3
                or out.shape[0] != 1
                or out.shape[2] != 6
                or out.role not in {None, "detections"}
            ):
                raise _invalid(
                    "End-to-end output must be [1,N,6] pixel xyxy, score and class ID."
                )
        elif decoder == "yolo-distance":
            sizes = {}
            for out in outputs:
                channels = {"boxes": 4, "scores": nc}.get(out.role)
                if (
                    channels is None
                    or out.layout != "NCHW"
                    or len(out.shape) != 4
                    or out.shape[0] != 1
                    or out.shape[1] != channels
                    or out.shape[2] != out.shape[3]
                    or shape[1] % out.shape[2]
                ):
                    raise _invalid(
                        "Distance YOLO requires explicit NCHW box/class branches."
                    )
                sizes.setdefault(out.shape[2], []).append(out.role)
            if any(sorted(roles) != ["boxes", "scores"] for roles in sizes.values()):
                raise _invalid(
                    "Each distance scale needs one box and one class branch."
                )
        elif decoder == "yolo-dfl":
            if post.get("reg_max") != 16 or nc == 64:
                raise _invalid(
                    "Kit DFL requires reg_max 16 and a class count other than 64."
                )
            sizes = {}
            for out in outputs:
                role = out.role
                if role is None:
                    role = (
                        "boxes"
                        if len(out.shape) >= 2 and out.shape[1] == 64
                        else "scores"
                    )
                expected_channels = {"boxes": 64, "scores": nc, "score_sum": 1}.get(
                    role
                )
                if (
                    out.layout != "NCHW"
                    or len(out.shape) != 4
                    or out.shape[0] != 1
                    or out.shape[2] != out.shape[3]
                    or out.shape[1] != expected_channels
                    or shape[1] % out.shape[2]
                ):
                    raise _invalid(
                        "DFL requires square NCHW box/class branches at exact input strides."
                    )
                sizes.setdefault(out.shape[2], []).append(role)
            if any(
                sorted(roles)
                not in (["boxes", "scores"], ["boxes", "score_sum", "scores"])
                for roles in sizes.values()
            ):
                raise _invalid(
                    "Each DFL feature size needs exactly one box and one class branch, optionally one explicit score_sum auxiliary output."
                )
        else:
            raise _invalid("Unsupported RKNN output decoder.")
        memory_mb = value.get("memory_mb", 64)
        if type(memory_mb) is not int or not 1 <= memory_mb <= 1024:
            raise _invalid("memory_mb must be an integer in 1..1024.")
        return ModelMetadata(
            model_id,
            path,
            sha,
            tuple(labels),
            input_spec,
            outputs,
            decoder,
            post["scores"],
            memory_mb,
            padding,
            post.get("objectness", False) if decoder == "yolo-decoded" else False,
            apply_nms,
            topk,
        )

    def list_models(self):
        """List package descriptors without reading full model blobs."""
        if not self.root.is_dir():
            return []
        result = []
        for path in sorted(self.root.rglob("model.json")):
            model_id = path.parent.relative_to(self.root).as_posix()
            try:
                result.append(self.get(model_id, verify=False).describe())
            except EdgeError as exc:
                result.append({"model_id": model_id, "error": exc.as_dict()})
        return result


class RegisteredModelStore(LocalModelStore):
    """Snapshot of exact packages granted by appmgr at process startup."""

    def __init__(self, root, packages):
        super().__init__(root)
        self.packages = {item["model_id"]: Path(item["directory"]) for item in packages}
        self.validation = {}
        self._checking = False

    def _model_dir(self, model_id):
        # Validate the ID even when it comes from a system binding.
        bundled = super()._model_dir(model_id)
        if model_id in self.packages:
            report = self.validation.get(model_id)
            if not self._checking and (not report or report["status"] != "ready"):
                raise EdgeError(
                    "This model has not passed device validation; open App Center model management.",
                    code="model_validation_required",
                    status_code=409,
                )
            return self.packages[model_id]
        return bundled

    def list_models(self):
        items = super().list_models()
        for identifier in self.packages:
            if self.validation.get(identifier, {}).get("status") == "ready":
                items.append(self.get(identifier, verify=False).describe())
        return items

    def validate(self, manager):
        """Real broker-authorized RKNN load and inference; keep only one session."""
        import numpy as np

        self._checking = True
        try:
            for identifier in self.packages:
                digest = None
                try:
                    metadata = self.get(identifier)
                    digest = metadata.sha256
                    manager.infer(
                        identifier,
                        np.zeros(
                            (metadata.input_size, metadata.input_size, 3),
                            dtype=np.uint8,
                        ),
                    )
                    self.validation[identifier] = {"status": "ready", "sha256": digest}
                except Exception as exc:
                    self.validation[identifier] = {
                        "status": "validation_failed",
                        "sha256": digest,
                        "error": getattr(exc, "code", "rknn_validation_failed"),
                    }
                finally:
                    manager.clear()
        finally:
            self._checking = False


class InferenceResult(dict):
    """JSON response with the small Pydantic method surface used by Workflows."""

    def model_dump(self, *, by_alias=False, exclude_none=False, **kwargs):
        return dict(self)


class EdgeModelManager:
    """Serialize load/infer/release, keeping at most one model session alive."""

    def __init__(self, store: LocalModelStore, backend_factory=None):
        self.store = store
        if backend_factory is None:
            from inference.edge.backend import KitBackend

            backend_factory = KitBackend
        self._backend_factory = backend_factory
        self._backend = None
        self._metadata = None
        self._lock = threading.RLock()

    def add_model(self, model_id, api_key=None, **kwargs):
        # api_key is a compatibility argument, never an NPU permission grant.
        with self._lock:
            if self._metadata is not None and self._metadata.model_id == model_id:
                return
            metadata = self.store.get(model_id)
            self.clear()
            self._backend = self._backend_factory(metadata)
            self._metadata = metadata

    def infer(self, model_id, image, **kwargs):
        """Infer BGR HWC uint8 images, with lists executed serially as batch 1."""
        with self._lock:
            self.add_model(model_id)
            if isinstance(image, list):
                if not image or len(image) > 8:
                    raise EdgeError(
                        "Image batches must contain 1..8 images.", code="invalid_input"
                    )
                return [
                    InferenceResult(self._backend.infer(item, **kwargs))
                    for item in image
                ]
            return InferenceResult(self._backend.infer(image, **kwargs))

    def infer_from_request_sync(self, model_id=None, request=None, **kwargs):
        if request is None and model_id is not None and not isinstance(model_id, str):
            request, model_id = model_id, None
        if request is None:
            raise EdgeError("An inference request is required.", code="invalid_input")

        def read(name, default=None):
            return (
                request.get(name, default)
                if isinstance(request, Mapping)
                else getattr(request, name, default)
            )

        request_id = read("model_id")
        if model_id is not None and request_id is not None and request_id != model_id:
            raise EdgeError(
                "Request model_id does not match the selected model.",
                code="invalid_model_id",
            )
        model_id = model_id or request_id

        def unwrap(image):
            if isinstance(image, Mapping):
                kind, value = image.get("type"), image.get("value")
            elif hasattr(image, "type"):
                kind, value = image.type, image.value
            else:
                return image
            if kind != "numpy":
                raise EdgeError(
                    "Decode request images to type=numpy before inference.",
                    code="invalid_input",
                )
            return value

        image = read("image")
        image = [unwrap(v) for v in image] if isinstance(image, list) else unwrap(image)
        options = {}
        for name in (
            "confidence",
            "iou_threshold",
            "max_detections",
            "class_filter",
            "class_agnostic_nms",
            "max_candidates",
        ):
            value = read(name)
            if value is not None:
                options[name] = value
        # The edge backend cannot silently promise unsupported preprocessing.
        if read("visualize_predictions", False):
            raise EdgeError(
                "Request visualization through a Workflow visualization block.",
                code="unsupported_option",
            )
        # These legacy controls alter pixels and therefore cannot be silently
        # ignored by a backend whose preprocessing is fixed in model metadata.
        for name in (
            "disable_preproc_auto_orient",
            "disable_preproc_contrast",
            "disable_preproc_grayscale",
            "disable_preproc_static_crop",
        ):
            if read(name, False):
                raise EdgeError(
                    f"{name} is not supported by this RKNN package.",
                    code="unsupported_option",
                )
        options.update(kwargs)
        return self.infer(model_id, image, **options)

    def remove(self, model_id, **kwargs):
        with self._lock:
            if self._metadata is not None and self._metadata.model_id == model_id:
                self.clear()

    def clear(self):
        with self._lock:
            if self._backend is None:
                return
            # If release raises, retain the handle so callers can retry and
            # cannot load a second session while release is uncertain.
            self._backend.release()
            self._backend = None
            self._metadata = None

    close = clear

    def list_models(self):
        with self._lock:
            loaded = self._metadata.model_id if self._metadata else None
            return [
                dict(item, loaded=item["model_id"] == loaded)
                for item in self.store.list_models()
            ]

    list = list_models

    def describe_models(self):
        with self._lock:
            return [] if self._metadata is None else [self._metadata.describe()]

    def get_task_type(self, model_id, api_key=None):
        self.store.get(model_id, verify=False)
        return "object-detection"
