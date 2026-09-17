"""Original Inference SDK video HTTP protocol over the device video runner."""

from uuid import uuid4

from fastapi import APIRouter, Request

from .errors import EdgeError


def invalid(message):
    raise EdgeError(message, code="invalid_pipeline", status_code=422)


def workflow_parameters(config, definitions, api_key=None):
    if (
        not isinstance(config, dict)
        or config.get("type", "WorkflowConfiguration") != "WorkflowConfiguration"
    ):
        invalid("processing_configuration must be WorkflowConfiguration")
    specification = config.get("workflow_specification")
    if specification is None:
        if not config.get("workflow_id"):
            invalid("Provide a workflow specification or workflow_id")
        specification = definitions.resolve(
            config.get("workspace_name", "local"),
            config["workflow_id"],
            api_key=api_key,
            use_cache=True,
            workflow_version_id=config.get("workflow_version_id"),
        )
    elif config.get("workflow_id") or config.get("workspace_name"):
        invalid("Provide either specification or a saved workflow identifier")
    parameters = config.get("workflows_parameters") or {}
    if not isinstance(parameters, dict):
        invalid("workflows_parameters must be an object")
    return {
        "specification": specification,
        "image_input": config.get("image_input_name", "image"),
        "inputs": parameters,
        "video_metadata_input_name": config.get(
            "video_metadata_input_name", "video_metadata"
        ),
        "disable_sinks": bool(config.get("disable_sinks", False)),
        "api_key": api_key,
    }


def command(identifier=None, **fields):
    return {
        "status": "success",
        "context": {"pipeline_id": identifier, "request_id": uuid4().hex},
        **fields,
    }


async def optional_body(request):
    raw = await request.body()
    if not raw:
        return {}
    try:
        value = await request.json()
    except ValueError:
        invalid("Expected a JSON object")
    if not isinstance(value, dict):
        invalid("Expected a JSON object")
    return value


def request_api_key(body, request, definitions):
    value = body.get("api_key")
    if value == getattr(getattr(definitions, "settings", None), "api_token", None):
        value = None
    return value or getattr(request.state, "roboflow_api_key", None)


def create_streaming_router(video, *, control, definitions):
    router = APIRouter(tags=["video"])

    def check(pipeline_id):
        rtc = getattr(video, "webrtc", None)
        if rtc and rtc.session and rtc.session.id == pipeline_id:
            return rtc.session
        if not pipeline_id or pipeline_id != video.status()["pipeline_id"]:
            raise EdgeError(
                "Unknown pipeline", code="pipeline_not_found", status_code=404
            )
        return video

    @router.post("/inference_pipelines/initialise")
    async def initialise(request: Request):
        body = await optional_body(request)

        def start():
            if "video_configuration" not in body:
                # Compatibility with the first device release's local controls.
                allowed = {
                    "model_id",
                    "specification",
                    "image_input",
                    "inputs",
                    "max_fps",
                    "video_reference",
                    "video_metadata_input_name",
                    "disable_sinks",
                    "api_key",
                }
                if set(body) - allowed:
                    invalid("Unknown pipeline fields")
                return video.start(
                    **{**body, "api_key": request_api_key(body, request, definitions)}
                )
            config = body.get("video_configuration")
            if (
                not isinstance(config, dict)
                or config.get("type") != "VideoConfiguration"
            ):
                invalid("video_configuration must be VideoConfiguration")
            reference = config.get("video_reference", 0)
            if isinstance(reference, list):
                if len(reference) != 1:
                    invalid("This device executes one video source at a time")
                reference = reference[0]
            if isinstance(reference, bool) or not isinstance(reference, (str, int)):
                invalid("video_reference must be a camera index, URL or video path")
            sink = body.get("sink_configuration") or {"type": "MemorySinkConfiguration"}
            if (
                not isinstance(sink, dict)
                or sink.get("type") != "MemorySinkConfiguration"
            ):
                invalid("This endpoint uses MemorySinkConfiguration")
            size = sink.get("results_buffer_size", 64)
            if type(size) is not int or size < 1:
                invalid("results_buffer_size must be positive")
            options = workflow_parameters(
                body.get("processing_configuration"),
                definitions,
                request_api_key(body, request, definitions),
            )
            result = video.start(
                **options,
                video_reference=reference,
                video_source_properties=config.get("video_source_properties"),
                max_fps=config.get("max_fps"),
                results_buffer_size=size,
            )
            return command(
                result["pipeline_id"],
                pipeline_id=result["pipeline_id"],
                results_buffer_size=min(2, size),
            )

        return await control(start)

    @router.get("/inference_pipelines/list")
    async def list_pipelines():
        state = video.status()
        rtc = getattr(video, "webrtc", None)
        if rtc and rtc.session:
            state = rtc.session.status()
        ids = (
            [state["pipeline_id"]]
            if state["pipeline_id"] and state["status"] != "stopped"
            else []
        )
        return command(pipelines=ids, pipeline_states=[state] if ids else [])

    @router.get("/inference_pipelines/{pipeline_id}/status")
    async def status(pipeline_id: str):
        state = check(pipeline_id).status()
        return command(
            pipeline_id,
            report={
                **state,
                "state": state["status"],
                "sources_metadata": [{"source_id": 0, **(state["last_frame"] or {})}],
            },
        )

    @router.get("/inference_pipelines/{pipeline_id}/consume")
    async def consume(pipeline_id: str, request: Request):
        body = await optional_body(request)
        excluded = body.get("excluded_fields") or request.query_params.getlist(
            "excluded_fields"
        )
        if not isinstance(excluded, list) or any(
            not isinstance(field, str) for field in excluded
        ):
            invalid("excluded_fields must be a list of field names")

        def read():
            target = check(pipeline_id)
            record = target.consume_one()
            if record is None:
                return command(pipeline_id, outputs=[], frames_metadata=[])
            result = record["result"]
            outputs = result if isinstance(result, list) else [result]
            outputs = [
                {name: value for name, value in output.items() if name not in excluded}
                for output in outputs
            ]
            metadata = {
                name: record[name]
                for name in ("frame_id", "source_id", "frame_timestamp")
            }
            return command(
                pipeline_id,
                outputs=outputs,
                frames_metadata=[metadata for _ in outputs],
            )

        return await control(read)

    def register_operation(name):
        async def operate(pipeline_id: str):
            target = check(pipeline_id)
            if target is not video and name == "terminate":
                await target.owner.close(pipeline_id)
                return command(pipeline_id)

            def action():
                target = check(pipeline_id)
                getattr(target, "stop" if name == "terminate" else name)()
                return command(pipeline_id)

            return await control(action)

        router.add_api_route(
            f"/inference_pipelines/{{pipeline_id}}/{name}", operate, methods=["POST"]
        )

        async def legacy(request: Request):
            body = await optional_body(request)
            target = check(body.get("pipeline_id"))
            if target is not video and name == "terminate":
                await target.owner.close(target.id)
                return target.status()

            def action():
                target = check(body.get("pipeline_id"))
                getattr(target, "stop" if name == "terminate" else name)()
                return target.status()

            return await control(action)

        router.add_api_route(f"/inference_pipelines/{name}", legacy, methods=["POST"])

    for name in ("pause", "resume", "terminate"):
        register_operation(name)

    @router.post("/inference_pipelines/consume")
    async def legacy_consume(request: Request):
        body = await optional_body(request)

        def read():
            target = check(body.get("pipeline_id"))
            return {"results": target.results()}

        return await control(read)

    return router
