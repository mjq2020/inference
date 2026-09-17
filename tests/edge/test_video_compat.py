import asyncio
import base64
import copy
import json
import os
import struct
import time
from types import SimpleNamespace

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from inference.edge.errors import EdgeError
from inference.edge.settings import EdgeSettings
from inference.edge.streaming import create_streaming_router
from inference.edge.video import VideoRunner
from inference.edge.webrtc import DiskVideoUpload, EdgeWebRTC

SPEC = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [],
    "outputs": [],
}


class Manager:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class Workflows:
    def __init__(self):
        self.calls = []
        self.cleared = []

    def validate(self, spec):
        assert spec == SPEC

    def preflight_models(self, spec, inputs=None):
        assert not any("model_id" in step for step in spec["steps"])

    def run(self, spec, inputs, **kwargs):
        self.calls.append((inputs["image"].shape, kwargs))
        return [{"count": len(self.calls), "extra": "omit me"}]

    def clear(self, stream_id=None):
        self.cleared.append(stream_id)


class Source:
    def __init__(self):
        self.closed = False

    def frames(self):
        for i in range(20):
            if self.closed:
                break
            time.sleep(0.025)
            yield SimpleNamespace(
                data=np.zeros((16, 24, 3), np.uint8), fmt="RGB", pts=i / 40
            )

    def cancel(self):
        self.closed = True

    def close(self):
        self.closed = True


async def control(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def make_service(tmp_path):
    settings = EdgeSettings(storage_root=tmp_path, max_fps=60)
    manager, workflows, source = Manager(), Workflows(), Source()
    video = VideoRunner(manager, workflows, settings, source_factory=lambda: source)
    definitions = SimpleNamespace(resolve=lambda *a, **k: SPEC)
    return settings, manager, workflows, source, video, definitions


def test_original_sdk_video_protocol_and_local_named_workflow(tmp_path):
    settings, manager, workflows, source, video, definitions = make_service(tmp_path)
    app = FastAPI()
    app.include_router(
        create_streaming_router(video, control=control, definitions=definitions)
    )

    @app.exception_handler(EdgeError)
    async def error(request, exc):
        return JSONResponse({"error": exc.code}, status_code=exc.status_code)

    body = {
        "video_configuration": {
            "type": "VideoConfiguration",
            "video_reference": [0],
            "max_fps": 40,
        },
        "processing_configuration": {
            "type": "WorkflowConfiguration",
            "workspace_name": "local",
            "workflow_id": "example",
            "video_metadata_input_name": "metadata",
        },
        "sink_configuration": {
            "type": "MemorySinkConfiguration",
            "results_buffer_size": 64,
        },
    }
    try:
        with TestClient(app) as client:
            result = client.post("/inference_pipelines/initialise", json=body)
            assert result.status_code == 200, result.text
            pid = result.json()["context"]["pipeline_id"]
            assert result.json()["status"] == "success"
            assert client.get("/inference_pipelines/list").json()["pipelines"] == [pid]
            deadline = time.monotonic() + 2
            while video.status()["frames"] < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
            response = client.request(
                "GET",
                f"/inference_pipelines/{pid}/consume",
                json={"excluded_fields": ["extra"]},
            )
            assert response.json()["outputs"][0]["count"] >= 1
            assert "extra" not in response.json()["outputs"][0]
            assert response.json()["frames_metadata"][0]["source_id"] == 0
            assert workflows.calls[0][1]["video_metadata_input_name"] == "metadata"
            assert (
                client.post(f"/inference_pipelines/{pid}/pause").json()["status"]
                == "success"
            )
            assert (
                client.get(f"/inference_pipelines/{pid}/status").json()["report"][
                    "state"
                ]
                == "paused"
            )
            assert client.post(f"/inference_pipelines/{pid}/resume").status_code == 200
            assert (
                client.post(f"/inference_pipelines/{pid}/terminate").status_code == 200
            )
            assert client.get("/inference_pipelines/missing/status").status_code == 404
    finally:
        video.stop()
    assert source.closed and manager.cleared


def test_video_upload_streams_to_disk_and_rejects_out_of_order(tmp_path):
    upload = DiskVideoUpload(tmp_path, 32)
    path = upload.path
    assert not upload.add(struct.pack("<II", 0, 2) + b"one")
    assert upload.add(struct.pack("<II", 1, 2) + b"two")
    assert open(path, "rb").read() == b"onetwo"
    upload.close()
    assert not os.path.exists(path)
    upload = DiskVideoUpload(tmp_path, 3)
    try:
        with pytest.raises(EdgeError):
            upload.add(struct.pack("<II", 1, 2) + b"x")
        with pytest.raises(EdgeError) as exc:
            upload.add(struct.pack("<II", 0, 1) + b"oversized")
        assert exc.value.status_code == 413
    finally:
        upload.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_real_webrtc_offer_video_data_and_release(tmp_path, legacy):
    rtc = pytest.importorskip("aiortc")
    av = pytest.importorskip("av")

    async def scenario():
        settings, manager, workflows, source, video, definitions = make_service(
            tmp_path
        )
        settings = SimpleNamespace(
            **{**settings.__dict__, "max_video_output_pixels": 96}
        )
        original_run = workflows.run

        def visualized(spec, inputs, **kwargs):
            import cv2

            output = original_run(spec, inputs, **kwargs)
            image = np.zeros((16, 24, 3), np.uint8)
            image[:, :, 1] = 255
            success, encoded = cv2.imencode(".jpg", image)
            assert success
            output[0]["preview"] = {
                "type": "base64",
                "value": base64.b64encode(encoded).decode(),
            }
            output[0]["predictions"] = {
                "image": {"width": 48, "height": 32},
                "predictions": [{"x": 24, "y": 16, "width": 12, "height": 8}],
            }
            return output

        workflows.run = visualized
        service = EdgeWebRTC(manager, workflows, settings, video, definitions, control)
        peer = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        messages, chunks = [], {}
        incoming = asyncio.Queue()
        channel = peer.createDataChannel("inference")

        @channel.on("message")
        def message(data):
            if legacy:
                messages.append({"serialized_output_data": {"count": json.loads(data)}})
                return
            frame_id, index, total = struct.unpack("<III", data[:12])
            chunk_map = chunks.setdefault(frame_id, {})
            chunk_map[index] = data[12:]
            if len(chunk_map) == total:
                messages.append(
                    json.loads(b"".join(chunk_map[i] for i in range(total)))
                )
                del chunks[frame_id]

        @peer.on("track")
        def track_received(track):
            incoming.put_nowait(track)

        class Track(rtc.VideoStreamTrack):
            async def recv(self):
                pts, time_base = await self.next_timestamp()
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((32, 48, 3), np.uint8), format="bgr24"
                )
                frame.pts, frame.time_base = pts, time_base
                return frame

        peer.addTrack(Track())
        try:
            await peer.setLocalDescription(await peer.createOffer())
            body = {
                "workflow_configuration": {
                    "type": "WorkflowConfiguration",
                    "workflow_specification": SPEC,
                },
                "webrtc_offer": {
                    "type": peer.localDescription.type,
                    "sdp": peer.localDescription.sdp,
                },
                "data_output": ["count"] if legacy else ["count", "predictions"],
                "stream_output": ["preview"],
                "declared_fps": 10,
            }
            import httpx

            from inference.edge.webrtc import create_webrtc_router

            app = FastAPI()
            router = create_webrtc_router(
                manager, workflows, settings, video, definitions, control
            )
            service = router.runtime
            app.include_router(router)
            app.include_router(
                create_streaming_router(video, control=control, definitions=definitions)
            )
            if legacy:
                body["processing_configuration"] = body.pop("workflow_configuration")
                body["video_configuration"] = {
                    "type": "VideoConfiguration",
                    "video_reference": 0,
                    "max_fps": 10,
                }
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                response = await client.post(
                    (
                        "/inference_pipelines/initialise_webrtc"
                        if legacy
                        else "/initialise_webrtc_worker"
                    ),
                    json=body,
                )
                assert response.status_code == 200, response.text
                answer = response.json()
            await peer.setRemoteDescription(
                rtc.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            track = await asyncio.wait_for(incoming.get(), 10)
            frame = await asyncio.wait_for(track.recv(), 10)
            assert (frame.width, frame.height) == (12, 8)
            deadline = time.monotonic() + 10
            while not messages and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert messages and messages[0]["serialized_output_data"]["count"] >= 1
            if not legacy:
                assert messages[0]["video_metadata"]["width"] == 48
                predictions = messages[0]["serialized_output_data"]["predictions"]
                assert predictions["image"] == {"width": 48, "height": 32}
                assert predictions["predictions"][0]["x"] == 24
                assert workflows.calls[0][0] == (32, 48, 3)
            assert video.is_active()
            assert service.session.incoming.qsize() <= 1
            assert service.session.outgoing.qsize() <= 1
            service.session.pause()
            await asyncio.sleep(0.2)
            count = service.session.frames
            await asyncio.sleep(0.15)
            assert service.session.frames == count
            service.session.resume()
        finally:
            await service.close()
            await peer.close()
        assert not video.is_active()
        assert manager.cleared == 1
        assert workflows.cleared

    asyncio.run(scenario())


def write_video(path, count=5):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
        for index in range(count):
            image = np.full((32, 48, 3), index * 30, np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="bgr24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_file_video_preserves_frames_and_independent_metadata(tmp_path):
    from inference.edge.video import AVSource

    path = tmp_path / "sample.mp4"
    write_video(path)
    settings, manager, workflows, _, _, _ = make_service(tmp_path)
    video = VideoRunner(manager, workflows, settings)
    try:
        video.start(specification=SPEC, video_reference=str(path), max_fps=40)
        video._thread.join(3)
        assert video.status()["status"] == "stopped", video.status()
        assert video.status()["frames"] == 5
        assert all(
            call[1]["frame_metadata"]["comes_from_video_file"]
            for call in workflows.calls
        )
        assert len(video.results()) == 2
    finally:
        video.stop()
    with pytest.raises(EdgeError) as exc:
        AVSource(str(path), max_pixels=100)
    assert exc.value.status_code == 413


def test_real_webrtc_uploaded_file_data_only_and_cleanup(tmp_path):
    rtc = pytest.importorskip("aiortc")
    path = tmp_path / "input.mp4"
    write_video(path, count=4)

    async def scenario():
        settings, manager, workflows, _, video, definitions = make_service(tmp_path)
        service = EdgeWebRTC(manager, workflows, settings, video, definitions, control)
        peer = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        data, upload = peer.createDataChannel("inference"), peer.createDataChannel(
            "video_upload"
        )
        opened, complete = asyncio.Event(), asyncio.Event()
        outputs = []
        chunks = {}

        @upload.on("open")
        def ready():
            opened.set()

        @data.on("message")
        def receive(message):
            frame_id, index, total = struct.unpack("<III", message[:12])
            parts = chunks.setdefault(frame_id, {})
            parts[index] = message[12:]
            if len(parts) != total:
                return
            result = json.loads(b"".join(parts[i] for i in range(total)))
            del chunks[frame_id]
            if result.get("processing_complete"):
                complete.set()
            else:
                outputs.append(result)

        try:
            await peer.setLocalDescription(await peer.createOffer())
            answer = await service.initialise(
                {
                    "workflow_configuration": {
                        "type": "WorkflowConfiguration",
                        "workflow_specification": SPEC,
                    },
                    "webrtc_offer": {
                        "type": peer.localDescription.type,
                        "sdp": peer.localDescription.sdp,
                    },
                    "stream_output": [],
                    "data_output": ["count"],
                    "webrtc_realtime_processing": False,
                }
            )
            await peer.setRemoteDescription(
                rtc.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            await asyncio.wait_for(opened.wait(), 10)
            upload.send(struct.pack("<II", 0, 1) + path.read_bytes())
            await asyncio.wait_for(complete.wait(), 10)
            assert len(outputs) == 4
            assert [item["serialized_output_data"]["count"] for item in outputs] == [
                1,
                2,
                3,
                4,
            ]
        finally:
            await service.close()
            await peer.close()
        assert not list((tmp_path / "video-uploads").glob("*"))
        assert not video.is_active()

    asyncio.run(scenario())


def test_webrtc_kit_cancellation_awaits_native_read_and_release_retry(tmp_path):
    import threading

    rtc = pytest.importorskip("aiortc")

    class FiniteSource:
        def __init__(self):
            self.cancelled = threading.Event()
            self.entered = threading.Event()
            self.borrowed = False
            self.closed = False

        def frames(self):
            while not self.cancelled.is_set():
                self.borrowed = True
                self.entered.set()
                self.cancelled.wait(0.05)
                self.borrowed = False
                if self.cancelled.is_set():
                    return
                yield SimpleNamespace(
                    data=np.zeros((32, 48, 3), np.uint8),
                    fmt="RGB",
                    pts=time.monotonic(),
                )

        def cancel(self):
            self.cancelled.set()

        def close(self):
            assert not self.borrowed
            self.closed = True

    async def scenario():
        settings, manager, workflows, _, video, definitions = make_service(tmp_path)
        source = FiniteSource()
        video.source_factory = lambda: source
        service = EdgeWebRTC(manager, workflows, settings, video, definitions, control)
        peer = rtc.RTCPeerConnection(rtc.RTCConfiguration(iceServers=[]))
        peer.addTransceiver("video", direction="recvonly")
        peer.createDataChannel("inference")
        original_clear = manager.clear

        def clear():
            if not manager.cleared:
                manager.cleared += 1
                raise RuntimeError("temporary release error")
            original_clear()

        manager.clear = clear
        try:
            await peer.setLocalDescription(await peer.createOffer())
            await service.initialise(
                {
                    "workflow_configuration": {
                        "type": "WorkflowConfiguration",
                        "workflow_specification": SPEC,
                    },
                    "webrtc_offer": {
                        "type": peer.localDescription.type,
                        "sdp": peer.localDescription.sdp,
                    },
                    "rtsp_url": "kit://camera",
                    "stream_output": None,
                }
            )
            assert await asyncio.to_thread(source.entered.wait, 2)
            with pytest.raises(EdgeError) as exc:
                await service.close()
            assert exc.value.status_code == 503
            assert source.closed and not source.borrowed
            assert video.is_active()
            await service.close()
            assert not video.is_active()
        finally:
            await service.close()
            await peer.close()

    asyncio.run(scenario())


def test_cloud_key_request_separates_device_and_roboflow_credentials():
    from inference.edge.streaming import request_api_key

    request = SimpleNamespace(state=SimpleNamespace(roboflow_api_key="cloud-key"))
    definitions = SimpleNamespace(settings=SimpleNamespace(api_token="device-key"))
    assert (
        request_api_key({"api_key": "device-key"}, request, definitions) == "cloud-key"
    )
    assert request_api_key({}, request, definitions) == "cloud-key"
    assert (
        request_api_key({"api_key": "other-cloud-key"}, request, definitions)
        == "other-cloud-key"
    )


def test_video_freezes_children_and_credentials_before_opening_source(tmp_path):
    settings, manager, workflows, source, video, _ = make_service(tmp_path)
    events = []
    frozen = copy.deepcopy(SPEC)

    def resolve(spec, api_key=None):
        events.append(("resolve", api_key))
        assert spec is SPEC
        return frozen

    def validate(spec, api_key=None):
        events.append(("validate", api_key))
        assert spec is frozen

    def run(spec, inputs, **kwargs):
        events.append(("run", kwargs.get("api_key")))
        assert spec is frozen
        return [{"count": 1}]

    def open_source():
        events.append(("open", None))
        return source

    workflows.resolve_definition = resolve
    workflows.validate = validate
    workflows.run = run
    video.source_factory = open_source
    try:
        video.start(specification=SPEC, api_key="cloud-key")
        deadline = time.monotonic() + 2
        while not video.status()["frames"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert events[:3] == [
            ("resolve", "cloud-key"),
            ("validate", "cloud-key"),
            ("open", None),
        ]
        assert ("run", "cloud-key") in events
    finally:
        video.stop()

    events.clear()

    def unresolved(spec, api_key=None):
        raise EdgeError("Unresolved child", code="invalid_workflow", status_code=422)

    workflows.resolve_definition = unresolved
    with pytest.raises(EdgeError):
        video.start(specification=SPEC, api_key="cloud-key")
    assert not events
