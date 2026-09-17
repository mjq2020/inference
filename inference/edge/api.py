"""Inference HTTP and original Workflow APIs for the RKNN-only profile."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from threading import BoundedSemaphore, RLock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from inference.core.entities.requests.workflows import (
    PredefinedWorkflowDescribeInterfaceRequest,
    PredefinedWorkflowInferenceRequest,
    WorkflowSpecificationDescribeInterfaceRequest,
    WorkflowSpecificationInferenceRequest,
)
from inference.runtime import IS_RV1126B

from .browser import install_browser
from .builder import create_builder_router
from .conversion import ConversionService
from .deployment import WorkflowDeployment
from .errors import EdgeError
from .images import decode_image
from .limits import encode_bounded_json
from .models import EdgeModelManager, LocalModelStore
from .settings import EdgeSettings
from .storage import WorkflowStore
from .video import VideoRunner
from .workflow_definitions import WorkflowDefinitions
from .workflows import EdgeWorkflows


class BudgetMiddleware:
    def __init__(self, app, settings):
        self.app, self.settings, self.active = app, settings, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] in ("GET", "HEAD") and (
            scope["path"] in ("/", "/ui/", "/build", "/build/")
            or scope["path"].startswith(("/ui/assets/", "/build/edit/"))
        ):
            return await self.app(scope, receive, send)
        if self.active >= self.settings.max_pending_requests:
            return await JSONResponse({"error": "request_queue_full"}, 429)(
                scope, receive, send
            )
        self.active += 1
        try:
            body = bytearray()
            deadline = asyncio.get_running_loop().time() + 15
            while True:
                try:
                    message = await asyncio.wait_for(
                        receive(),
                        timeout=max(0, deadline - asyncio.get_running_loop().time()),
                    )
                except asyncio.TimeoutError:
                    return await JSONResponse({"error": "request_body_timeout"}, 408)(
                        scope, receive, send
                    )
                if message["type"] == "http.disconnect":
                    return
                if (
                    len(body) + len(message.get("body", b""))
                    > self.settings.max_request_bytes
                ):
                    return await JSONResponse({"error": "request_too_large"}, 413)(
                        scope, receive, send
                    )
                body.extend(message.get("body", b""))
                more_body = message.get("more_body", False)
                del message
                if not more_body:
                    break
            headers = dict(scope.get("headers", []))
            if (
                body
                and not any(
                    key in headers
                    for key in (b"authorization", b"x-inference-token", b"x-csrf")
                )
                and b"application/json" in headers.get(b"content-type", b"")
            ):
                try:
                    payload = json.loads(body)
                    if isinstance(payload, dict) and isinstance(
                        payload.get("api_key"), str
                    ):
                        scope.setdefault("state", {})["legacy_api_key"] = payload[
                            "api_key"
                        ]
                    del payload
                except (ValueError, RecursionError):
                    pass
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    payload = bytes(body)
                    body.clear()
                    return {"type": "http.request", "body": payload, "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        finally:
            self.active -= 1


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    image: object
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    max_detections: int = Field(default=100, ge=1, le=1000)
    class_filter: list[str] | None = None
    class_agnostic_nms: bool = False
    api_key: str | None = None


def create_app(
    settings=None,
    *,
    manager=None,
    source_factory=None,
    converter=None,
    on_result=None,
    workflow_fetcher=None,
):
    if not IS_RV1126B:
        raise RuntimeError(
            "Launch with inference_edge.py or set INFERENCE_RUNTIME_PROFILE=rv1126b before import"
        )
    settings = settings or EdgeSettings.from_env()
    manager = manager or EdgeModelManager(LocalModelStore(settings.model_root))
    workflows = EdgeWorkflows(manager, settings)
    video = VideoRunner(
        manager, workflows, settings, source_factory=source_factory, on_result=on_result
    )
    conversion = ConversionService(converter=converter)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="edge-http")
    control_lock = RLock()
    work_slots = BoundedSemaphore(settings.max_pending_requests)

    class BoundedJSONResponse(JSONResponse):
        def render(self, content):
            return encode_bounded_json(content, settings.max_response_bytes)

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.deployment.start_background()
            yield
        finally:
            try:
                await app.state.deployment.close()
            finally:
                try:
                    await app.state.webrtc.close()
                finally:
                    try:
                        video.stop()
                    finally:
                        pool.shutdown(wait=True, cancel_futures=True)
                        if not video.is_active():
                            workflows.clear()
                            manager.clear()

    app = FastAPI(
        title="Inference for RV1126B",
        version="0.3.1",
        lifespan=lifespan,
        default_response_class=BoundedJSONResponse,
    )
    install_browser(app, settings)
    app.add_middleware(BudgetMiddleware, settings=settings)
    app.state.manager, app.state.workflows, app.state.video = manager, workflows, video
    app.state.settings = settings

    async def execute(fn, *args, **kwargs):
        if not work_slots.acquire(blocking=False):
            raise EdgeError(
                "Device work queue is full", code="request_queue_full", status_code=429
            )
        try:
            future = pool.submit(partial(fn, *args, **kwargs))
        except BaseException:
            work_slots.release()
            raise
        # Count the actual job until it ends, even if its HTTP caller disconnects.
        future.add_done_callback(lambda _: work_slots.release())
        return await asyncio.wrap_future(future)

    async def control(fn, *args, idle=False, **kwargs):
        def operation():
            with control_lock:
                if idle and video.is_active():
                    raise EdgeError(
                        "Stop the camera pipeline before HTTP inference or changing models",
                        code="pipeline_active",
                        status_code=409,
                    )
                return fn(*args, **kwargs)

        return await execute(operation)

    @app.exception_handler(EdgeError)
    async def edge_error(_request, exc):
        return JSONResponse(
            getattr(exc, "workflow_response", {"error": exc.code, "message": str(exc)}),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def request_error(_request, exc):
        # Do not echo base64 uploads or large invalid values into error bodies.
        details = [
            {"location": item["loc"], "type": item["type"]}
            for item in exc.errors()[:10]
        ]
        return JSONResponse(
            {
                "error": "invalid_request",
                "details": details,
                "detail": [
                    {"loc": item["loc"], "type": item["type"], "msg": item["msg"]}
                    for item in exc.errors()[:10]
                ],
            },
            status_code=422,
        )

    @app.get("/healthz")
    async def health():
        return {
            "status": "ok",
            "profile": "rv1126b",
            "backend": "rknn",
            "training": False,
            "model_conversion": "remote",
        }

    @app.get("/capabilities")
    async def capabilities():
        return {
            "platform": "rv1126b",
            "runtime": "2.3.2",
            "model_formats": ["rknn"],
            "tasks": ["object-detection"],
            "max_active_models": 1,
            "max_pipelines": 1,
            "model_batch_size": 1,
            "workflow_batching": True,
            "image_inputs": ["base64", "url"],
            "conversion": conversion.status(),
            "training": False,
            "cpu_model_fallback": False,
        }

    @app.get("/workflow-deployment")
    async def deployment_status():
        return app.state.deployment.status()

    @app.get("/model/registry")
    async def registry():
        return {"models": await execute(manager.list_models)}

    @app.post("/model/add")
    async def add_model(body: dict):
        model_id = body.get("model_id")
        if not isinstance(model_id, str):
            raise EdgeError(
                "model_id is required", code="invalid_model_id", status_code=422
            )
        await control(manager.add_model, model_id, idle=True)
        return {"models": await execute(manager.list_models)}

    @app.post("/model/remove")
    async def remove_model(body: dict):
        await control(manager.remove, body.get("model_id"), idle=True)
        return {"status": "ok"}

    @app.post("/model/clear")
    async def clear_models():
        await control(manager.clear, idle=True)
        return {"status": "ok"}

    @app.post("/infer/object_detection")
    async def detect(body: DetectionRequest):
        def infer():
            image = decode_image(
                body.image,
                max_pixels=settings.max_image_pixels,
                max_bytes=settings.max_request_bytes,
            )
            params = body.model_dump(exclude={"image", "model_id", "api_key"})
            return manager.infer(body.model_id, image, **params)

        return await control(infer, idle=True)

    @app.get("/workflows/execution_engine/versions")
    async def versions():
        from inference.core.workflows.execution_engine.core import (
            get_available_versions,
        )

        return {"versions": get_available_versions()}

    @app.get("/workflows/blocks/describe")
    @app.post("/workflows/blocks/describe")
    async def blocks(body: dict | None = None):
        body = body or {}
        result = await execute(
            workflows.describe,
            execution_engine_version=body.get("execution_engine_version"),
            dynamic_blocks_definitions=body.get("dynamic_blocks_definitions"),
        )
        return metadata_response(result)

    @app.post("/workflows/run")
    @app.post("/infer/workflows", deprecated=True)
    async def run_workflow(
        body: WorkflowSpecificationInferenceRequest, request: Request
    ):
        return await run_request(body, body.specification, cloud_key(request, body))

    async def run_request(body, specification, api_key=None):
        return await control(
            workflows.run_request,
            specification,
            body.inputs,
            enable_profiling=body.enable_profiling,
            debug=body.debug,
            is_preview=getattr(body, "is_preview", False),
            disable_sinks=body.disable_sinks,
            workflow_id=body.workflow_id,
            excluded_fields=body.excluded_fields,
            api_key=api_key,
            idle=True,
        )

    def cloud_key(request, body):
        explicit = body.api_key or request.query_params.get("api_key")
        if explicit == settings.api_token:
            explicit = None
        return explicit or getattr(request.state, "roboflow_api_key", None)

    async def resolve(request, body, workspace, workflow_id):
        return await execute(
            definitions.resolve,
            workspace,
            workflow_id,
            api_key=cloud_key(request, body),
            use_cache=body.use_cache,
            workflow_version_id=body.workflow_version_id,
        )

    @app.post("/{workspace_name}/workflows/{workflow_id}")
    @app.post("/infer/workflows/{workspace_name}/{workflow_id}", deprecated=True)
    async def run_named_workflow(
        workspace_name: str,
        workflow_id: str,
        body: PredefinedWorkflowInferenceRequest,
        request: Request,
    ):
        specification = await resolve(request, body, workspace_name, workflow_id)
        if not body.workflow_id:
            body = body.model_copy(update={"workflow_id": workflow_id})
        return await run_request(body, specification, cloud_key(request, body))

    @app.post("/workflows/describe_interface")
    async def describe_interface(
        body: WorkflowSpecificationDescribeInterfaceRequest, request: Request
    ):
        return metadata_response(
            await execute(
                workflows.describe_interface,
                body.specification,
                cloud_key(request, body),
            )
        )

    @app.post("/{workspace_name}/workflows/{workflow_id}/describe_interface")
    async def describe_named_interface(
        workspace_name: str,
        workflow_id: str,
        body: PredefinedWorkflowDescribeInterfaceRequest,
        request: Request,
    ):
        specification = await resolve(request, body, workspace_name, workflow_id)
        return metadata_response(
            await execute(
                workflows.describe_interface, specification, cloud_key(request, body)
            )
        )

    @app.get("/model-conversion")
    async def conversion_status():
        return conversion.status()

    @app.post("/model-conversion")
    async def conversion_request():
        raise EdgeError(
            "The ONNX-to-RKNN HTTP protocol has not been configured",
            code="converter_unconfigured",
            status_code=503,
        )

    store = WorkflowStore(settings.storage_root / "workflows")
    definitions = WorkflowDefinitions(store, settings, fetcher=workflow_fetcher)
    workflows.definitions = definitions
    app.state.workflow_definitions = definitions
    app.state.workflow_store = store
    app.state.deployment = WorkflowDeployment(settings, definitions, video, control)

    @app.get("/app-center/models")
    async def prepared_models():
        return {"validation": getattr(manager.store, "validation", {})}

    @app.get("/app-center/workflow-runtime")
    async def workflow_runtime(include_result: bool = False, include_overlay: bool = False):
        # App Center authenticates with the existing device credential. Never
        # return RTSP credentials, input parameters or another session's output.
        deployment = app.state.deployment.status()
        state = video.status()
        owned = bool(deployment["pipeline_id"] and deployment["pipeline_id"] == state["pipeline_id"])
        return {
            "deployment": deployment,
            "video": state if owned else None,
            "latest": video.latest_result() if owned and include_result else None,
            "preview": video.preview.status() if owned and state["status"] == "running" else None,
            "overlay": video.preview.overlay(include_image=settings.video_source == "rtsp")
            if owned and state["status"] == "running" and include_overlay else None,
            "service_only": not settings.workflow_autostart,
        }

    @app.get("/app-center/workflow-preview")
    async def workflow_preview(request: Request, pipeline_id: str, output: str):
        deployment = app.state.deployment.status()
        state = video.status()
        if (not deployment["pipeline_id"] or deployment["pipeline_id"] != pipeline_id
                or state["pipeline_id"] != pipeline_id or state["status"] != "running"):
            raise EdgeError("Workflow preview is not running", code="preview_inactive", status_code=409)
        code, data, content_type, headers = await execute(
            video.preview.read, pipeline_id, output, request.headers.get("if-none-match"))
        return Response(content=data, status_code=code, media_type=content_type, headers=headers)
    from .builder import metadata_response

    app.include_router(
        create_builder_router(workflows, manager, store, execute=execute)
    )
    from .streaming import create_streaming_router

    app.include_router(
        create_streaming_router(video, control=control, definitions=definitions)
    )
    from .webrtc import create_webrtc_router

    webrtc_router = create_webrtc_router(
        manager, workflows, settings, video, definitions, control
    )
    app.state.webrtc = webrtc_router.runtime
    app.include_router(webrtc_router)
    return app
