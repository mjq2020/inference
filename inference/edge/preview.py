"""Bounded, read-only views of a running Workflow's existing image outputs."""

import base64
import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock

from .errors import EdgeError
from .overlay import extract_overlays


class WorkflowPreview:
    MAX_OUTPUTS = 16
    ORIGINAL_LEASE_SECONDS = 3

    def __init__(self, max_bytes):
        self.max_bytes = min(max_bytes, 4 * 1024 * 1024)
        self._lock = Lock()
        self.reset(None)

    def reset(self, pipeline_id, fps=3):
        with self._lock:
            self.pipeline_id = pipeline_id
            self._max_age = max(10, 3 / fps)
            self._outputs = {}
            self._known = {}
            self._encoded = OrderedDict()
            self._original = None
            self._original_until = 0
            self._frame_id = None
            self._timestamp = None
            self._updated = 0
            self._overlays = {}
            self._known_overlays = {}
            self._size = (0, 0)

    def capture_original(self, bgr):
        """Encode before the graph can mutate its input, only while requested."""
        with self._lock:
            wanted = time.monotonic() < self._original_until
        if not wanted:
            return None
        import cv2

        height, width = bgr.shape[:2]
        scale = min(1, (1280 * 720 / (height * width)) ** 0.5)
        image = (
            bgr
            if scale == 1
            else cv2.resize(
                bgr, (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok or encoded.nbytes > self.max_bytes:
            return None
        return encoded.tobytes()

    def publish(self, record, original, size=(0, 0), root_outputs=None, image_input=None):
        # References reuse the runner's bounded serialized result, without
        # copying image strings or decoding every image for every frame.
        outputs = {}
        visited = 0

        def visit(value, path, label, depth=0):
            nonlocal visited
            visited += 1
            if visited > 256 or depth > 6 or len(outputs) >= self.MAX_OUTPUTS:
                return
            if isinstance(value, dict):
                if value.get("type") == "base64" and isinstance(
                    value.get("value"), str
                ):
                    identifier = hashlib.sha256(json.dumps(path).encode()).hexdigest()[
                        :20
                    ]
                    outputs[identifier] = (label[:256], value["value"])
                # Only explicit Workflow image values and image lists are
                # exposed; detection metadata and arbitrary nested JSON aren't.
            elif isinstance(value, list):
                for index, item in enumerate(value[: self.MAX_OUTPUTS]):
                    visit(item, [*path, index], f"{label}[{index}]", depth + 1)

        batches = record.get("result")
        if isinstance(batches, dict):
            batches = [batches]
        for batch_index, batch in enumerate((batches or [])[: self.MAX_OUTPUTS]):
            if not isinstance(batch, dict):
                continue
            for name, value in list(batch.items())[:128]:
                label = (
                    str(name) if len(batches) == 1 else f"{name} [{batch_index + 1}]"
                )
                visit(value, [batch_index, name], label)
        overlays = extract_overlays(record.get("result"), size, root_outputs, image_input)
        with self._lock:
            if record["pipeline_id"] != self.pipeline_id:
                return
            self._outputs = outputs
            self._overlays = overlays
            self._size = size
            for identifier, group in overlays.items():
                if (
                    len(self._known_overlays) < self.MAX_OUTPUTS
                    or identifier in self._known_overlays
                ):
                    self._known_overlays[identifier] = group["name"]
            for identifier, (label, _) in outputs.items():
                if len(self._known) < self.MAX_OUTPUTS or identifier in self._known:
                    self._known[identifier] = label
            self._encoded.clear()
            self._original = (
                original if time.monotonic() < self._original_until else None
            )
            self._frame_id = record["frame_id"]
            self._timestamp = record["frame_timestamp"]
            self._updated = time.monotonic()

    def status(self):
        with self._lock:
            if time.monotonic() >= self._original_until:
                self._original = None
            return {
                "version": 1,
                "pipeline_id": self.pipeline_id,
                "outputs": [
                    {"id": key, "name": label, "available": key in self._outputs}
                    for key, label in self._known.items()
                ],
                "overlay_outputs": [
                    {"id": key, "name": label}
                    for key, label in self._known_overlays.items()
                ],
            }

    def overlay(self, include_image=False):
        """Return small geometry, optionally paired with the exact original frame."""
        with self._lock:
            if include_image:
                self._original_until = time.monotonic() + self.ORIGINAL_LEASE_SECONDS
            if (
                self._frame_id is None
                or time.monotonic() - self._updated > self._max_age
            ):
                return None
            data = {
                "pipeline_id": self.pipeline_id,
                "frame_id": self._frame_id,
                "timestamp": self._timestamp,
                "width": self._size[0],
                "height": self._size[1],
                "groups": list(self._overlays.values()),
            }
            if include_image:
                data["image"] = (
                    base64.b64encode(self._original).decode("ascii")
                    if self._original
                    else None
                )
            return data

    def read(self, pipeline_id, output, if_none_match=None):
        with self._lock:
            if not pipeline_id or pipeline_id != self.pipeline_id:
                raise EdgeError(
                    "Workflow preview has changed",
                    code="preview_changed",
                    status_code=409,
                )
            if output == "original":
                self._original_until = time.monotonic() + self.ORIGINAL_LEASE_SECONDS
            elif output not in self._known:
                raise EdgeError(
                    "Unknown image output",
                    code="preview_output_missing",
                    status_code=404,
                )
            if (
                self._frame_id is None
                or time.monotonic() - self._updated > self._max_age
            ):
                raise EdgeError(
                    "Waiting for a new Workflow result",
                    code="preview_waiting",
                    status_code=425,
                )
            value = (
                self._original if output == "original" else self._outputs.get(output)
            )
            if value is None:
                raise EdgeError(
                    "Waiting for this image output",
                    code="preview_waiting",
                    status_code=425,
                )
            tag = f'"{self.pipeline_id}:{self._frame_id}:{output}"'
            headers = {
                "Cache-Control": "no-store",
                "ETag": tag,
                "X-Workflow-Pipeline": self.pipeline_id,
                "X-Workflow-Frame": str(self._frame_id),
                "X-Workflow-Timestamp": self._timestamp,
            }
            if if_none_match == tag:
                return 304, b"", "image/jpeg", headers
            if output == "original":
                return 200, value, "image/jpeg", headers
            if output in self._encoded:
                data, content_type = self._encoded[output]
                self._encoded.move_to_end(output)
            else:
                encoded = value[1]
                if len(encoded) > (self.max_bytes + 2) // 3 * 4:
                    raise EdgeError(
                        "Preview image exceeds the size limit",
                        code="preview_too_large",
                        status_code=413,
                    )
                try:
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, UnicodeError) as exc:
                    raise EdgeError(
                        "Invalid preview image", code="preview_invalid", status_code=422
                    ) from exc
                content_type = (
                    "image/jpeg"
                    if data.startswith(b"\xff\xd8\xff")
                    else "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else None
                )
                if not content_type or len(data) > self.max_bytes:
                    raise EdgeError(
                        "Unsupported preview image",
                        code="preview_invalid",
                        status_code=422,
                    )
                self._encoded[output] = (data, content_type)
                while len(self._encoded) > 2:
                    self._encoded.popitem(last=False)
            return 200, data, content_type, headers
