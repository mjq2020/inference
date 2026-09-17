"""Real, in-process WebRTC transport for the original Workflow SDK protocol.

Media dependencies are lazy: health and HTTP inference do not load FFmpeg,
OpenSSL or SRTP. One session shares the device's video execution reservation.
"""

import asyncio
import json
import math
import struct
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Request

from .errors import EdgeError
from .images import decode_image
from .limits import encode_bounded_json
from .streaming import (
    command,
    invalid,
    optional_body,
    request_api_key,
    workflow_parameters,
)


def media_modules():
    try:
        import aiortc
        import av
        from aiortc.contrib.media import MediaPlayer
        from aiortc.mediastreams import MediaStreamError
    except (ImportError, OSError) as exc:
        raise EdgeError(
            "WebRTC requires working aiortc, PyAV and SRTP media packages",
            code="webrtc_dependencies_unavailable",
            status_code=503,
        ) from exc
    return aiortc, av, MediaPlayer, MediaStreamError


def is_image(value):
    return (
        isinstance(value, dict)
        and value.get("type") == "base64"
        and isinstance(value.get("value"), str)
    )


def preview_image(array, max_pixels=1280 * 720):
    """Resize only the outgoing video track; never mutate Workflow pixels/results."""
    import cv2

    height, width = array.shape[:2]
    if height * width <= max_pixels:
        return array
    scale = math.sqrt(max_pixels / (height * width))
    output_width = max(1, int(width * scale))
    output_height = max(1, int(height * scale))
    return cv2.resize(
        array, (output_width, output_height), interpolation=cv2.INTER_AREA
    )


class DiskVideoUpload:
    """Ordered original SDK upload protocol; payloads stream directly to disk."""

    def __init__(self, directory, max_bytes):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.file = tempfile.NamedTemporaryFile(
            dir=directory, suffix=".video", delete=False
        )
        self.path = self.file.name
        self.max_bytes = max_bytes
        self.total = None
        self.index = self.size = 0
        self.complete = False

    def add(self, message):
        if not isinstance(message, bytes) or len(message) <= 8 or self.complete:
            invalid("Invalid video upload chunk")
        index, total = struct.unpack("<II", message[:8])
        if (
            total < 1
            or total > self.max_bytes
            or index != self.index
            or (self.total is not None and total != self.total)
        ):
            invalid("Video chunks must arrive once in their declared order")
        self.total = total
        self.size += len(message) - 8
        if self.size > self.max_bytes:
            raise EdgeError(
                "Video upload exceeds the disk budget",
                code="video_upload_too_large",
                status_code=413,
            )
        self.file.write(memoryview(message)[8:])
        self.index += 1
        if self.index == total:
            self.file.close()
            self.complete = True
        return self.complete

    def close(self):
        self.file.close()
        Path(self.path).unlink(missing_ok=True)


class EdgeWebRTC:
    def __init__(self, manager, workflows, settings, video, definitions, control):
        self.manager, self.workflows, self.settings = manager, workflows, settings
        self.video, self.definitions, self.control = video, definitions, control
        video.webrtc = self
        self._lock = asyncio.Lock()
        self.session = None

    async def initialise(self, body, *, legacy=False):
        modules = media_modules()
        offer = body.get("webrtc_offer")
        if (
            not isinstance(offer, dict)
            or offer.get("type") != "offer"
            or not isinstance(offer.get("sdp"), str)
        ):
            invalid("A WebRTC SDP offer is required")
        async with self._lock:
            if self.session is not None:
                raise EdgeError(
                    "One WebRTC session is already active",
                    code="pipeline_limit",
                    status_code=409,
                )
            options = await self.control(
                workflow_parameters,
                body.get("workflow_configuration"),
                self.definitions,
                body.get("api_key"),
            )
            options["specification"] = await self.control(
                self.video.prepare_workflow,
                options["specification"],
                options.get("api_key"),
            )
            session = WebRTCSession(self, body, options, modules)
            session.legacy = legacy
            self.video.reserve(session.id)
            self.session = session
            try:
                await session.start(offer)
            except BaseException as exc:
                await session.close()
                self.session = None
                if isinstance(exc, (TypeError, ValueError)):
                    raise EdgeError(
                        "Invalid or unsupported WebRTC offer",
                        code="invalid_webrtc_offer",
                        status_code=422,
                    ) from exc
                raise
            return command(
                session.id,
                sdp=session.peer.localDescription.sdp,
                type=session.peer.localDescription.type,
                session_id=session.id,
            )

    async def close(self, session_id=None):
        async with self._lock:
            if self.session is None:
                return
            if session_id is not None and session_id != self.session.id:
                raise EdgeError(
                    "Unknown WebRTC session",
                    code="webrtc_session_not_found",
                    status_code=404,
                )
            await self.session.close()
            self.session = None

    def status(self):
        if self.session is None:
            return {"active": False}
        return {
            "active": True,
            "session_id": self.session.id,
            "frames": self.session.frames,
            "connection_state": self.session.peer.connectionState,
        }


class WebRTCSession:
    def __init__(self, owner, body, options, modules):
        self.owner, self.body, self.options = owner, body, options
        self.rtc, self.av, self.MediaPlayer, self.MediaStreamError = modules
        self.id = uuid4().hex
        self.frames = 0
        self._paused = False
        self._results = deque(maxlen=2)
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_frame = None
        self.legacy = False
        self.tasks = set()
        self.closed = False
        self.inflight = None
        self.source = self.player = self.upload = self.input_track = None
        self.channel = None
        self.incoming = asyncio.Queue(maxsize=1)
        self.outgoing = asyncio.Queue(maxsize=1)
        self.stream_output = body.get("stream_output")
        self.data_output = body.get("data_output") or []
        for value in (self.stream_output, self.data_output):
            if value is not None and (
                not isinstance(value, list)
                or any(not isinstance(x, str) for x in value)
            ):
                invalid("WebRTC output selections must be lists of names")
        fps = body.get("declared_fps") or owner.settings.max_fps
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (float, int))
            or not 0 < fps <= 240
        ):
            invalid("declared_fps must be a positive frame rate")
        self.fps = min(fps, owner.settings.max_fps)
        timeout = body.get("processing_timeout") or 1800
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not 0 < timeout <= 86400
        ):
            invalid("processing_timeout must be between 0 and 86400 seconds")
        self.timeout = timeout
        self.realtime = body.get("webrtc_realtime_processing", True)
        self.is_file = False
        config = body.get("webrtc_config") or {}
        if not isinstance(config, dict):
            invalid("webrtc_config must be an object")
        servers = config.get("iceServers", [])
        if body.get("webrtc_turn_config") and not servers:
            servers = [body["webrtc_turn_config"]]
        if not isinstance(servers, list) or len(servers) > 8:
            invalid("Invalid ICE server list")
        try:
            ice = [self.rtc.RTCIceServer(**server) for server in servers]
        except (TypeError, ValueError) as exc:
            invalid("Invalid ICE server configuration")
        # No hidden public STUN or TURN dependency; use only explicitly supplied servers.
        self.peer = self.rtc.RTCPeerConnection(
            self.rtc.RTCConfiguration(iceServers=ice)
        )
        session = self

        class OutputTrack(self.rtc.VideoStreamTrack):
            async def recv(track):
                value = await session.outgoing.get()
                if value is None:
                    raise session.MediaStreamError
                return value

        self.output_track = OutputTrack() if self.stream_output != [] else None

    def task(self, coroutine):
        async def guarded():
            try:
                await coroutine
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                try:
                    await self.fail(str(exc))
                finally:
                    if not self.closed:
                        asyncio.create_task(self.owner.close(self.id))

        task = asyncio.create_task(guarded())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def status(self):
        return {
            "pipeline_id": self.id,
            "status": (
                "stopped" if self.closed else "paused" if self._paused else "running"
            ),
            "frames": self.frames,
            "error": None,
            "retained_results": len(self._results),
            "started_at": self._started_at,
            "last_frame": self._last_frame,
            "results_buffer_size": 2,
        }

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def consume_one(self):
        try:
            return self._results.popleft()
        except IndexError:
            return None

    def results(self):
        result = list(self._results)
        self._results.clear()
        return result

    async def start(self, offer):
        @self.peer.on("track")
        def track_received(track):
            if track.kind != "video":
                track.stop()
                return
            if self.input_track is not None or self.source is not None:
                track.stop()
                return
            self.input_track = track
            self.task(self.read_track(track))

        @self.peer.on("connectionstatechange")
        def connection_changed():
            if self.peer.connectionState in ("failed", "closed") and not self.closed:
                asyncio.create_task(self.owner.close(self.id))

        @self.peer.on("datachannel")
        def channel_received(channel):
            if channel.label == "video_upload":

                @channel.on("message")
                def receive_upload(message):
                    if self.closed:
                        return
                    if isinstance(message, bytes) and len(message) <= 1:
                        channel.send(message)
                        return
                    try:
                        if self.upload is None:
                            self.upload = DiskVideoUpload(
                                self.owner.settings.storage_root / "video-uploads",
                                self.owner.settings.max_request_bytes * 32,
                            )
                        if self.upload.add(message):
                            self.is_file = True
                            self.task(self.open_player(self.upload.path))
                    except Exception as exc:
                        self.task(self.fail(str(exc)))

                return
            self.channel = channel

            @channel.on("message")
            def update(message):
                try:
                    if not isinstance(message, str) or len(message) > 16384:
                        return
                    data = json.loads(message)
                    if not isinstance(data, dict):
                        return
                    for key in ("stream_output", "data_output"):
                        if (
                            key in data
                            and isinstance(data[key], list)
                            and all(isinstance(x, str) for x in data[key])
                        ):
                            setattr(self, key, data[key])
                except ValueError:
                    return

        if self.output_track is not None:
            self.peer.addTrack(self.output_track)
        await self.peer.setRemoteDescription(self.rtc.RTCSessionDescription(**offer))
        await self.peer.setLocalDescription(await self.peer.createAnswer())
        self.task(self.process())
        self.task(self.deadline())
        reference = self.body.get("rtsp_url") or self.body.get("mjpeg_url")
        if reference:
            if reference in ("kit://camera", "camera"):
                self.source = await self.open_native_source(
                    self.owner.video.open_source, "kit://camera"
                )
                self.task(self.read_camera())
            else:
                await self.open_player(reference)

    async def open_player(self, reference):
        from urllib.parse import urlsplit

        if self.input_track is not None or self.source is not None:
            invalid("Only one video source is supported per session")
        if not self.is_file and urlsplit(str(reference)).scheme not in (
            "rtsp",
            "rtsps",
            "http",
            "https",
        ):
            invalid("Use an RTSP/HTTP source or kit://camera")
        from .video import AVSource

        self.source = await self.open_native_source(
            AVSource, reference, self.owner.settings.max_image_pixels
        )
        self.is_file = self.source.is_file
        self.task(self.read_camera())

    async def open_native_source(self, factory, *args):
        pending = asyncio.create_task(self.owner.control(factory, *args))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                source = await pending
            except Exception:
                pass
            else:
                await asyncio.to_thread(source.close)
            raise

    async def enqueue(self, frame):
        if frame.width * frame.height > self.owner.settings.max_image_pixels:
            raise EdgeError(
                "Video frame exceeds pixel budget",
                code="image_too_large",
                status_code=413,
            )
        if self.is_file and not self.realtime:
            await self.incoming.put(frame)
        else:
            if self.incoming.full():
                self.incoming.get_nowait()
            self.incoming.put_nowait(frame)

    async def read_track(self, track):
        try:
            while not self.closed:
                await self.enqueue(await track.recv())
        except self.MediaStreamError:
            await self.incoming.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.fail(str(exc))

    async def read_camera(self):
        iterator = self.source.frames()
        pending = None
        clock_start = pts_start = None

        def next_frame():
            return next(iterator, None)

        try:
            while not self.closed:
                pending = asyncio.create_task(asyncio.to_thread(next_frame))
                item = await asyncio.shield(pending)
                pending = None
                if item is None:
                    await self.incoming.put(None)
                    return
                if self.is_file and self.realtime:
                    pts = getattr(item, "pts", 0)
                    if clock_start is None:
                        clock_start, pts_start = time.monotonic(), pts
                    await asyncio.sleep(
                        max(0, clock_start + pts - pts_start - time.monotonic())
                    )
                frame = self.av.VideoFrame.from_ndarray(
                    item.data,
                    format="rgb24" if getattr(item, "fmt", "RGB") == "RGB" else "bgr24",
                )
                frame.pts = int(getattr(item, "pts", time.monotonic()) * 90000)
                frame.time_base = Fraction(1, 90000)
                await self.enqueue(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.fail(str(exc))
        finally:
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            if hasattr(iterator, "close"):
                await asyncio.to_thread(iterator.close)

    async def send(self, message, frame_id):
        channel = self.channel
        if channel is None or channel.readyState != "open":
            return
        if self.legacy:
            data = message.get("serialized_output_data") or {}
            if not self.data_output or self.data_output[0] not in data:
                return
            value = data[self.data_output[0]]
            if is_image(value):
                return
            payload = encode_bounded_json(
                value, min(48 * 1024, self.owner.settings.max_response_bytes)
            )
            if channel.bufferedAmount <= self.owner.settings.max_response_bytes * 2:
                channel.send(payload.decode("utf-8"))
            return
        payload = encode_bounded_json(message, self.owner.settings.max_response_bytes)
        limit = self.owner.settings.max_response_bytes * 2
        if channel.bufferedAmount > limit:
            if self.realtime:
                return
            deadline = time.monotonic() + 15
            while channel.bufferedAmount > limit and time.monotonic() < deadline:
                if channel.readyState != "open":
                    return
                await asyncio.sleep(0.01)
            if channel.bufferedAmount > limit:
                raise EdgeError(
                    "WebRTC receiver stopped consuming results",
                    code="webrtc_receiver_stalled",
                    status_code=503,
                )
        count = (len(payload) + 49151) // 49152
        for index in range(count):
            if channel.readyState != "open":
                return
            channel.send(
                struct.pack("<III", frame_id, index, count)
                + payload[index * 49152 : (index + 1) * 49152]
            )
            await asyncio.sleep(0)

    async def process(self):
        next_frame = 0.0
        try:
            while not self.closed:
                frame = await self.incoming.get()
                if frame is None:
                    await self.send(
                        {
                            "serialized_output_data": None,
                            "video_metadata": None,
                            "errors": [],
                            "processing_complete": True,
                            "termination_reason": "video_ended",
                        },
                        self.frames + 1,
                    )
                    await asyncio.sleep(0.1)
                    break
                while self._paused and not self.closed:
                    await asyncio.sleep(0.05)
                if self.closed:
                    break
                await asyncio.sleep(max(0, next_frame - time.monotonic()))
                next_frame = time.monotonic() + 1 / self.fps
                timestamp = datetime.now(timezone.utc)
                bgr = frame.to_ndarray(format="bgr24")
                self.frames += 1
                metadata = {
                    "frame_number": self.frames,
                    "frame_timestamp": timestamp,
                    "fps": self.fps,
                    "comes_from_video_file": self.is_file,
                }
                self.inflight = asyncio.create_task(
                    asyncio.to_thread(
                        self.owner.workflows.run,
                        self.options["specification"],
                        {**self.options["inputs"], self.options["image_input"]: bgr},
                        stream_id=self.id,
                        frame_metadata=metadata,
                        video_metadata_input_name=self.options[
                            "video_metadata_input_name"
                        ],
                        disable_sinks=self.options["disable_sinks"],
                        api_key=self.options.get("api_key"),
                    )
                )
                result = await asyncio.shield(self.inflight)
                self.inflight = None
                output = result[0] if isinstance(result, list) and result else result
                if not isinstance(output, dict):
                    invalid("Workflow must return an output object")
                record = {
                    "pipeline_id": self.id,
                    "frame_id": self.frames,
                    "frame_timestamp": timestamp.isoformat(),
                    "source_id": 0,
                    "pts_us": int((frame.time or 0) * 1e6),
                    "result": [output],
                }
                encode_bounded_json(record, self.owner.settings.max_response_bytes)
                self._results.append(record)
                self._last_frame = {
                    "width": frame.width,
                    "height": frame.height,
                    "frame_timestamp": timestamp.isoformat(),
                }
                errors = []
                fields = (
                    [key for key, value in output.items() if not is_image(value)]
                    if self.data_output == ["*"]
                    else self.data_output
                )
                data = {name: output[name] for name in fields if name in output}
                errors.extend(
                    f"Output '{name}' not found"
                    for name in fields
                    if name not in output
                )
                await self.send(
                    {
                        "serialized_output_data": data or None,
                        "video_metadata": {
                            "frame_id": self.frames,
                            "received_at": timestamp.isoformat(),
                            "pts": frame.pts,
                            "time_base": (
                                float(frame.time_base) if frame.time_base else None
                            ),
                            "declared_fps": self.fps,
                            "width": frame.width,
                            "height": frame.height,
                        },
                        "errors": errors,
                        "processing_complete": False,
                        "termination_reason": None,
                    },
                    self.frames,
                )
                if self.output_track is not None:
                    selected = next(
                        (
                            output[name]
                            for name in (self.stream_output or output.keys())
                            if name in output and is_image(output[name])
                        ),
                        None,
                    )
                    array = (
                        decode_image(
                            selected, max_pixels=self.owner.settings.max_image_pixels
                        )
                        if selected is not None
                        else bgr
                    )
                    outgoing = self.av.VideoFrame.from_ndarray(
                        preview_image(
                            array,
                            getattr(
                                self.owner.settings,
                                "max_video_output_pixels",
                                1280 * 720,
                            ),
                        ),
                        format="bgr24",
                    )
                    outgoing.pts = (
                        frame.pts
                        if frame.pts is not None
                        else int(time.monotonic() * 90000)
                    )
                    outgoing.time_base = frame.time_base or Fraction(1, 90000)
                    if self.outgoing.full():
                        self.outgoing.get_nowait()
                    self.outgoing.put_nowait(outgoing)
                del frame, bgr, result, output
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send(
                {
                    "serialized_output_data": None,
                    "video_metadata": None,
                    "errors": [str(exc)[:2000]],
                    "processing_complete": True,
                    "termination_reason": "error",
                },
                self.frames,
            )
        finally:
            if not self.closed:
                asyncio.create_task(self.owner.close(self.id))

    async def fail(self, message):
        await self.send(
            {
                "serialized_output_data": None,
                "video_metadata": None,
                "errors": [message[:2000]],
                "processing_complete": True,
                "termination_reason": "error",
            },
            self.frames,
        )
        asyncio.create_task(self.owner.close(self.id))

    async def deadline(self):
        await asyncio.sleep(self.timeout)
        await self.fail("Processing time limit reached")

    async def close(self):
        self.closed = True
        if self.source:
            self.source.cancel()
        if self.input_track:
            self.input_track.stop()
        if self.output_track:
            self.output_track.stop()
        current = asyncio.current_task()
        pending = [task for task in self.tasks if task is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if self.inflight:
            await asyncio.gather(self.inflight, return_exceptions=True)
            self.inflight = None
        await self.peer.close()
        if self.source:
            await asyncio.to_thread(self.source.close)
        if self.upload:
            self.upload.close()
        await asyncio.to_thread(self.owner.workflows.clear, stream_id=self.id)
        # Retain the reservation if platform release fails; a repeated close retries.
        try:
            await asyncio.to_thread(self.owner.manager.clear)
        except Exception as exc:
            raise EdgeError(
                "Model release failed; close the WebRTC session again to retry",
                code="pipeline_stop_failed",
                status_code=503,
            ) from exc
        self.owner.video.release(self.id)
        self._results.clear()
        for queue in (self.incoming, self.outgoing):
            while not queue.empty():
                queue.get_nowait()
        self.outgoing.put_nowait(None)


def create_webrtc_router(manager, workflows, settings, video, definitions, control):
    router = APIRouter(tags=["webrtc"])
    router.runtime = EdgeWebRTC(
        manager, workflows, settings, video, definitions, control
    )

    @router.post("/initialise_webrtc_worker")
    async def initialise(request: Request):
        body = await optional_body(request)
        body["api_key"] = request_api_key(body, request, definitions)
        return await router.runtime.initialise(body)

    @router.get("/webrtc/status")
    async def status():
        return router.runtime.status()

    @router.post("/inference_pipelines/initialise_webrtc")
    async def legacy_initialise(request: Request):
        body = await optional_body(request)
        video_configuration = body.get("video_configuration")
        if (
            not isinstance(video_configuration, dict)
            or video_configuration.get("type") != "VideoConfiguration"
        ):
            invalid("video_configuration must be VideoConfiguration")
        converted = {
            **body,
            "api_key": request_api_key(body, request, definitions),
            "workflow_configuration": body.get("processing_configuration"),
            "declared_fps": video_configuration.get("max_fps")
            or body.get("webcam_fps"),
            # In the old endpoint processing_timeout is a per-frame queue wait,
            # not the worker lifetime. Do not terminate after its 5 ms default.
            "processing_timeout": 1800,
            "stream_output": body.get("stream_output") or None,
        }
        return await router.runtime.initialise(converted, legacy=True)

    @router.post("/webrtc/session/heartbeat")
    async def heartbeat(request: Request):
        body = await optional_body(request)
        state = router.runtime.status()
        if not state.get("active") or body.get("session_id") != state.get("session_id"):
            raise EdgeError(
                "Unknown WebRTC session",
                code="webrtc_session_not_found",
                status_code=404,
            )
        return command(body["session_id"])

    @router.post("/webrtc/session/end")
    async def end(request: Request):
        body = await optional_body(request)
        if not body.get("session_id"):
            invalid("session_id is required")
        await router.runtime.close(body["session_id"])
        return command(body["session_id"])

    return router
