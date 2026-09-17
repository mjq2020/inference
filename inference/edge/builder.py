"""Original Builder HTTP contracts without cloud model loaders or training code.

Authentication and per-session CSRF are enforced by the containing edge app.
Pass its bounded ``execute`` coroutine to keep blocking work off the HTTP loop.
"""

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .errors import EdgeError
from .limits import encode_bounded_json


def metadata_response(content):
    """Discovery schemas have a separate bound from per-frame inference output."""
    return Response(
        encode_bounded_json(content, 16 * 1024 * 1024), media_type="application/json"
    )


def create_builder_router(workflows, manager, store, *, execute=None):
    router = APIRouter()

    async def call(function, *args):
        try:
            if execute is not None:
                return await execute(function, *args)
            return function(*args)
        except OSError as exc:
            raise EdgeError(
                "Workflow storage is unavailable",
                code="workflow_storage_unavailable",
                status_code=503,
            ) from exc

    @router.get("/build/api")
    async def list_workflows():
        return await call(store.list_for_builder)

    @router.get("/build/api/models")
    async def list_models():
        def describe():
            from inference.core.workflows.core_steps.models.rknn.v1 import (
                BlockManifest,
                BlockManifestV2,
                BlockManifestV3,
            )

            compatible = [
                block_type
                for manifest in (BlockManifest, BlockManifestV2, BlockManifestV3)
                for block_type in manifest.model_fields["type"].annotation.__args__
            ]
            models = []
            for model in manager.list_models():
                if (
                    model.get("format") != "rknn"
                    or model.get("task_type") != "object-detection"
                ):
                    continue
                models.append(
                    {
                        **model,
                        "name": model.get("name") or model["model_id"],
                        "model_architecture": model.get("decoder", "rknn"),
                        "is_foundation": False,
                        "compatible_block_types": compatible,
                        "aliases": [],
                    }
                )
            # Display supported downloadable models in the original canvas.
            # Saving a reference schedules preparation in the system app manager.
            existing = {model["model_id"] for model in models}
            for identifier, name, architecture in (
                ("yolov8n-640", "YOLOv8 Nano · COCO", "yolov8"),
                ("yolov11n-640", "YOLO11 Nano · COCO", "yolov11"),
                ("yolo26n-640", "YOLO26 Nano · COCO", "yolo26"),
            ):
                if identifier not in existing:
                    models.append(
                        {
                            "model_id": identifier,
                            "name": name,
                            "task_type": "object-detection",
                            "model_architecture": architecture,
                            "is_foundation": False,
                            "compatible_block_types": compatible,
                            "aliases": [],
                            "preparation_required": True,
                        }
                    )
            return {"models": models}

        return await call(describe)

    @router.get("/build/api/{workflow_id}")
    async def get_workflow(workflow_id: str):
        return await call(store.get_for_builder, workflow_id)

    @router.post("/build/api/{workflow_id}", status_code=201)
    async def save_workflow(workflow_id: str, config: dict):
        return await call(store.save, workflow_id, config)

    @router.delete("/build/api/{workflow_id}")
    async def delete_workflow(workflow_id: str):
        return await call(store.delete, workflow_id)

    @router.get("/workflows/definition/schema")
    async def definition_schema():
        return metadata_response(await call(workflows.schema))

    @router.post("/workflows/validate")
    async def validate_workflow(specification: dict, request: Request):
        api_key = request.query_params.get("api_key") or getattr(
            request.state, "roboflow_api_key", None
        )
        await call(workflows.validate, specification, api_key)
        return {"status": "ok"}

    @router.post("/workflows/blocks/dynamic_outputs")
    async def dynamic_outputs(step_manifest: dict):
        return await call(workflows.dynamic_outputs, step_manifest)

    return router
