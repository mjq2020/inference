"""One on-demand video thread with bounded serialized results and native borrows."""

import math
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from .errors import EdgeError
from .limits import encode_bounded_json


def video_error(exc):
    # FFmpeg exceptions can include the authenticated input URL.
    return re.sub(r"rtsps?://[^\s'\"<>]+", "[RTSP source]", str(exc))[:2000]


class CancellableCameraSource:
    """Use finite native waits; every native borrow closes in the camera thread.

    Kit's frame iterator retries acquisition timeouts indefinitely. Its pinned
    converter is reused here while the public SDK ``acquire`` API supplies the
    cancellation point that a long-lived HTTP application needs.
    """

    def __init__(self, max_pixels=4096 * 2160):
        self.max_pixels = max_pixels
        from kit.adapters.official import OfficialFrameSource

        self._cancelled = Event()
        self._converter = OfficialFrameSource(
            timeout_ms=500, prefer_rga=True, verbose=False
        )

    def frames(self):
        from kit.adapters.frame_source import Frame
        from recamera_ext import AcquireTimeoutError, FrameSource

        with FrameSource(timeout_ms=500) as source:
            while not self._cancelled.is_set():
                try:
                    frame = source.acquire()
                except AcquireTimeoutError:
                    continue
                try:
                    if self._cancelled.is_set():
                        return
                    if int(frame.width) * int(frame.height) > self.max_pixels:
                        raise EdgeError(
                            "Camera image exceeds pixel budget",
                            code="image_too_large",
                            status_code=413,
                        )
                    rgb, _, _ = self._converter._convert(frame)
                    output = Frame(
                        data=rgb,
                        w=int(frame.width),
                        h=int(frame.height),
                        fmt="RGB",
                        pts=frame.pts_us / 1e6,
                    )
                finally:
                    # Conversion owns its RGB storage; return the DMA borrow
                    # before running model inference or yielding to the caller.
                    frame.release()
                yield output

    def cancel(self):
        self._cancelled.set()

    def close(self):
        self.cancel()
        self._converter.close()


def open_camera(max_pixels=4096 * 2160):
    if os.getenv("RECAMERA_FRAME_SOURCE") != "official":
        raise EdgeError(
            "Camera access requires an appmgr camera.frames allocation",
            code="camera_authorization_required",
            status_code=403,
        )
    return CancellableCameraSource(max_pixels=max_pixels)


class OpenCVSource:
    """Additional camera, file and RTSP/HTTP sources using installed media codecs.

    No GStreamer pipeline strings or shell commands are accepted as references.
    OpenCV builds without a working decoder fail explicitly at source opening.
    """

    def __init__(self, reference, max_pixels, properties=None):
        from pathlib import Path
        from urllib.parse import urlsplit

        import cv2

        if isinstance(reference, bool) or not isinstance(reference, (str, int)):
            raise EdgeError(
                "Invalid video reference", code="invalid_video_source", status_code=422
            )
        if isinstance(reference, str):
            scheme = urlsplit(reference).scheme.lower()
            if scheme and scheme not in ("http", "https", "rtsp", "rtsps"):
                raise EdgeError(
                    "Use an HTTP/RTSP URL or a video file",
                    code="invalid_video_source",
                    status_code=422,
                )
            if not scheme and not Path(reference).is_file():
                raise EdgeError(
                    "Video file does not exist",
                    code="video_source_not_found",
                    status_code=404,
                )
            self.is_file = not scheme
        else:
            self.is_file = False
        self.max_pixels = max_pixels
        self._cancelled = Event()
        self._capture = cv2.VideoCapture()
        # Supported FFmpeg/GStreamer backends apply finite network read waits.
        # Some V4L2 backends reject open parameters, so camera indices omit them.
        params = (
            []
            if isinstance(reference, int)
            else [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                2000,
            ]
        )
        if not self._capture.open(reference, cv2.CAP_ANY, params):
            self._capture.release()
            raise EdgeError(
                "Cannot open video source with the installed codecs",
                code="video_source_unavailable",
                status_code=503,
            )
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for name, value in (properties or {}).items():
                attribute = (
                    name
                    if str(name).startswith("CAP_PROP_")
                    else "CAP_PROP_" + str(name).upper()
                )
                prop = getattr(cv2, attribute, None)
                if (
                    prop is None
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    raise EdgeError(
                        "Invalid video source property",
                        code="invalid_video_source",
                        status_code=422,
                    )
                if not self._capture.set(prop, value):
                    raise EdgeError(
                        "Video source property is unsupported by its decoder",
                        code="invalid_video_source",
                        status_code=422,
                    )
        except Exception:
            self._capture.release()
            raise

    def frames(self):
        import cv2

        while not self._cancelled.is_set():
            ok, array = self._capture.read()
            if not ok:
                if self.is_file or self._cancelled.is_set():
                    return
                raise EdgeError(
                    "Video source stopped delivering frames",
                    code="video_source_lost",
                    status_code=503,
                )
            if array.shape[0] * array.shape[1] > self.max_pixels:
                raise EdgeError(
                    "Video frame exceeds pixel budget",
                    code="image_too_large",
                    status_code=413,
                )
            yield SimpleNamespace(
                data=array,
                fmt="BGR",
                pts=self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000,
            )

    def cancel(self):
        self._cancelled.set()

    def close(self):
        self.cancel()
        self._capture.release()


class AVSource:
    """Sequential FFmpeg decoding, without MediaPlayer's unbounded frame queue."""

    def __init__(self, reference, max_pixels):
        from pathlib import Path
        from urllib.parse import urlsplit

        import av

        if not isinstance(reference, str):
            raise EdgeError(
                "Invalid video reference", code="invalid_video_source", status_code=422
            )
        scheme = urlsplit(reference).scheme.lower()
        if scheme and scheme not in ("http", "https", "rtsp", "rtsps"):
            raise EdgeError(
                "Use an HTTP/RTSP URL or video file",
                code="invalid_video_source",
                status_code=422,
            )
        if not scheme and not Path(reference).is_file():
            raise EdgeError(
                "Video file does not exist",
                code="video_source_not_found",
                status_code=404,
            )
        self.is_file = not scheme
        self._cancelled = Event()
        self.max_pixels = max_pixels
        options = {"rtsp_transport": "tcp"} if scheme in ("rtsp", "rtsps") else {}
        try:
            self.container = av.open(reference, timeout=(5, 2), options=options)
            if not self.container.streams.video:
                self.container.close()
                raise ValueError("No video stream")
            self.stream = self.container.streams.video[0]
            self.stream.codec_context.thread_count = 2
            if self.stream.width * self.stream.height > max_pixels:
                self.container.close()
                raise EdgeError(
                    "Video frame exceeds pixel budget",
                    code="image_too_large",
                    status_code=413,
                )
        except EdgeError:
            raise
        except Exception as exc:
            raise EdgeError(
                "Cannot open video source with FFmpeg",
                code="video_source_unavailable",
                status_code=503,
            ) from exc

    def frames(self):
        for frame in self.container.decode(self.stream):
            if self._cancelled.is_set():
                return
            if frame.width * frame.height > self.max_pixels:
                raise EdgeError(
                    "Video frame exceeds pixel budget",
                    code="image_too_large",
                    status_code=413,
                )
            yield SimpleNamespace(
                data=frame.to_ndarray(format="bgr24"), fmt="BGR", pts=frame.time or 0
            )

    def cancel(self):
        self._cancelled.set()

    def close(self):
        self.cancel()
        self.container.close()


class VideoRunner:
    def __init__(
        self, manager, workflows, settings, *, source_factory=None, on_result=None
    ):
        self.manager, self.workflows, self.settings = manager, workflows, settings
        self.source_factory = source_factory
        self.on_result = on_result
        self._lock = Lock()
        self._stop = Event()
        self._paused = Event()
        self._thread = None
        self._source = None
        self._results = deque(maxlen=2)
        self._id = None
        self._error = None
        self._frames = 0
        self._model_cleanup_failed = False
        self._reference = 0
        self._started_at = None
        self._last_frame = None
        self._external_owner = None
        from .preview import WorkflowPreview

        self.preview = WorkflowPreview(settings.max_response_bytes)

    def reserve(self, owner):
        """Share the device execution slot with WebRTC without a second camera."""
        with self._lock:
            if self._external_owner or (self._thread and self._thread.is_alive()):
                raise EdgeError(
                    "A video session is already active",
                    code="pipeline_limit",
                    status_code=409,
                )
            if self._model_cleanup_failed:
                raise EdgeError(
                    "Retry pipeline stop to release its model",
                    code="pipeline_cleanup_required",
                    status_code=409,
                )
            self._external_owner = owner

    def release(self, owner):
        with self._lock:
            if self._external_owner == owner:
                self._external_owner = None

    def open_source(self, reference=0, properties=None):
        if self.source_factory:
            # Existing integrations supply a no-argument Kit source factory.
            if reference in (0, "0", "camera", "kit://camera"):
                return self.source_factory()
            return self.source_factory(reference, properties or {})
        if reference in (0, "0", "camera", "kit://camera"):
            return open_camera(max_pixels=self.settings.max_image_pixels)
        if isinstance(reference, str) and not properties:
            try:
                import av
            except ImportError:
                pass
            else:
                return AVSource(reference, self.settings.max_image_pixels)
        return OpenCVSource(reference, self.settings.max_image_pixels, properties)

    def prepare_workflow(self, specification, api_key=None):
        """Resolve children and validate the complete graph before opening media."""
        resolver = getattr(self.workflows, "resolve_definition", None)
        if callable(resolver):
            specification = resolver(specification, api_key=api_key)
        self.workflows.validate(
            specification, **({"api_key": api_key} if api_key else {})
        )
        return specification

    def start(
        self,
        *,
        model_id=None,
        specification=None,
        image_input="image",
        inputs=None,
        max_fps=None,
        video_reference=0,
        video_source_properties=None,
        video_metadata_input_name="video_metadata",
        disable_sinks=False,
        results_buffer_size=2,
        api_key=None,
    ):
        fps = self.settings.max_fps if max_fps is None else max_fps
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or not 0 < fps <= self.settings.max_fps
        ):
            raise EdgeError(
                "Requested frame rate exceeds device limit",
                code="fps_limit",
                status_code=422,
            )
        if bool(model_id) == bool(specification):
            raise EdgeError(
                "Specify exactly one model_id or workflow specification",
                code="invalid_pipeline",
                status_code=422,
            )
        with self._lock:
            if self._external_owner or (self._thread and self._thread.is_alive()):
                raise EdgeError(
                    "Only one camera pipeline may run",
                    code="pipeline_limit",
                    status_code=409,
                )
            if self._model_cleanup_failed:
                raise EdgeError(
                    "Stop the previous pipeline again to release its model",
                    code="pipeline_cleanup_required",
                    status_code=409,
                )
            if model_id:
                self.manager.add_model(model_id)
            else:
                specification = self.prepare_workflow(specification, api_key)
                self.workflows.preflight_models(specification, inputs)
                image_names = {
                    item.get("name")
                    for item in specification.get("inputs", [])
                    if item.get("type") in ("WorkflowImage", "InferenceImage")
                }
                if image_input not in image_names:
                    raise EdgeError(
                        "image_input must name a declared WorkflowImage",
                        code="invalid_pipeline",
                        status_code=422,
                    )
            try:
                source = self.open_source(video_reference, video_source_properties)
            except Exception:
                if model_id:
                    self.manager.clear()
                raise
            self._source = source
            self._id = uuid4().hex
            self.preview.reset(self._id, fps)
            self._error = None
            self._frames = 0
            self._results = deque(maxlen=max(1, min(2, results_buffer_size)))
            self._reference = video_reference
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._last_frame = None
            self._stop.clear()
            self._paused.clear()
            self._thread = Thread(
                target=self._run,
                args=(
                    source,
                    model_id,
                    specification,
                    image_input,
                    inputs or {},
                    fps,
                    video_metadata_input_name,
                    disable_sinks,
                    api_key,
                ),
                name="edge-camera",
                daemon=True,
            )
            self._thread.start()
            return {"pipeline_id": self._id, "status": "running"}

    def _run(
        self,
        source,
        model_id,
        specification,
        image_input,
        inputs,
        fps,
        video_metadata_input_name,
        disable_sinks,
        api_key,
    ):
        next_frame = 0.0
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                if getattr(source, "is_file", False):
                    while self._paused.is_set() and not self._stop.wait(0.05):
                        pass
                    if self._stop.wait(max(0, next_frame - time.monotonic())):
                        break
                now = time.monotonic()
                if self._paused.is_set() or now < next_frame:
                    continue
                next_frame = now + 1.0 / fps
                # OfficialFrameSource gives a privately owned RGB array. Copy
                # before leaving this iteration; no borrowed frame crosses threads.
                rgb = frame.data
                if (
                    not isinstance(rgb, np.ndarray)
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                    or rgb.dtype != np.uint8
                    or not rgb.size
                ):
                    raise EdgeError(
                        "Camera must provide an HWC uint8 RGB frame",
                        code="invalid_camera_frame",
                        status_code=422,
                    )
                if rgb.shape[0] * rgb.shape[1] > self.settings.max_image_pixels:
                    raise EdgeError(
                        "Camera image exceeds pixel budget",
                        code="image_too_large",
                        status_code=413,
                    )
                bgr = np.ascontiguousarray(
                    rgb if getattr(frame, "fmt", "RGB") == "BGR" else rgb[:, :, ::-1]
                )
                timestamp = datetime.now(timezone.utc)
                metadata = {
                    "frame_number": self._frames + 1,
                    "frame_timestamp": timestamp,
                    "fps": fps,
                    "comes_from_video_file": bool(getattr(source, "is_file", False)),
                }
                original = self.preview.capture_original(bgr)
                if specification:
                    result = self.workflows.run(
                        specification,
                        {**inputs, image_input: bgr},
                        stream_id=self._id,
                        frame_metadata=metadata,
                        video_metadata_input_name=video_metadata_input_name,
                        disable_sinks=disable_sinks,
                        api_key=api_key,
                    )
                else:
                    result = self.manager.infer(model_id, bgr)
                pts_us = getattr(frame, "pts_us", None)
                if pts_us is None:
                    pts_us = int(getattr(frame, "pts", 0) * 1e6)
                record = {
                    "pipeline_id": self._id,
                    "frame_id": self._frames,
                    "pts_us": pts_us,
                    "frame_timestamp": timestamp.isoformat(),
                    "source_id": 0,
                    "result": result,
                }
                # Bound retained output bytes, including any custom callback payload.
                encode_bounded_json(record, self.settings.max_response_bytes)
                root_outputs = (
                    None
                    if not specification
                    else {
                        output["name"]
                        for output in specification.get("outputs", [])
                        if output.get("coordinates_system", "parent") == "parent"
                    }
                )
                self.preview.publish(
                    record, original, (bgr.shape[1], bgr.shape[0]), root_outputs, image_input
                )
                if self.on_result:
                    self.on_result(record)
                with self._lock:
                    self._frames += 1
                    self._last_frame = {
                        "width": bgr.shape[1],
                        "height": bgr.shape[0],
                        "frame_timestamp": timestamp.isoformat(),
                    }
                    self._results.append(record)
                del bgr, rgb, result, record, original
        except Exception as exc:
            with self._lock:
                self._error = video_error(exc)
        finally:
            self.preview.reset(None)
            cleanup = (
                source.close,
                lambda: self.workflows.clear(stream_id=self._id),
            )
            for close in cleanup:
                try:
                    close()
                except Exception as exc:
                    with self._lock:
                        self._error = self._error or video_error(exc)
            try:
                self.manager.clear()
            except Exception as exc:
                with self._lock:
                    self._model_cleanup_failed = True
                    self._error = self._error or video_error(exc)

    def is_active(self):
        with self._lock:
            return bool(
                self._external_owner or (self._thread and self._thread.is_alive())
            )

    def status(self):
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            return {
                "pipeline_id": self._id,
                "status": (
                    "failed"
                    if self._error
                    else (
                        "paused"
                        if running and self._paused.is_set()
                        else "running" if running else "stopped"
                    )
                ),
                "frames": self._frames,
                "error": self._error,
                "retained_results": len(self._results),
                "started_at": self._started_at,
                "last_frame": self._last_frame,
                "results_buffer_size": self._results.maxlen,
            }

    def results(self):
        with self._lock:
            results = list(self._results)
            self._results.clear()
            return results

    def latest_result(self):
        """Read the existing bounded buffer without stealing SDK consumers' data."""
        with self._lock:
            return self._results[-1] if self._results else None

    def consume_one(self):
        with self._lock:
            return self._results.popleft() if self._results else None

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def stop(self):
        self._stop.set()
        thread = self._thread
        source = self._source
        if source is not None and callable(getattr(source, "cancel", None)):
            source.cancel()
        if thread:
            thread.join(timeout=5)
            if thread.is_alive():
                raise EdgeError(
                    "Camera worker did not stop; runtime must exit",
                    code="pipeline_stop_failed",
                    status_code=503,
                )
        with self._lock:
            if self._model_cleanup_failed:
                try:
                    self.manager.clear()
                except Exception as exc:
                    self._error = self._error or str(exc)
                    raise EdgeError(
                        "Model release failed; stop the pipeline again to retry",
                        code="pipeline_stop_failed",
                        status_code=503,
                    ) from exc
                self._model_cleanup_failed = False
            self._results.clear()
            self._source = None
        return self.status()
