"""Explicit startup selection of one saved Workflow, without editing its graph."""

import asyncio
import copy
import hashlib
import json
from threading import Event
from urllib.parse import urlsplit

from .errors import EdgeError
from .video import video_error


def select_single_image_input(specification):
    names = [
        item.get("name")
        for item in specification.get("inputs", [])
        if item.get("type") in ("WorkflowImage", "InferenceImage")
    ]
    if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
        raise EdgeError(
            "Workflow deployment requires exactly one declared image input",
            code="workflow_image_input_ambiguous",
            status_code=400,
        )
    return names[0]


class WorkflowDeployment:
    def __init__(self, settings, definitions, video, control):
        self.settings, self.definitions, self.video, self.control = (
            settings,
            definitions,
            video,
            control,
        )
        self._status = "pending" if settings.workflow_autostart else "disabled"
        self._pipeline_id = None
        self._digest = None
        self._error = None
        self._stopping = Event()
        self._task = None

    def start_background(self):
        if self.settings.workflow_autostart and self._task is None:
            self._task = asyncio.create_task(self.start(), name="workflow-autostart")

    async def close(self):
        # A cancelled await cannot stop a running executor thread. Let the
        # bounded resolver finish and observe this flag before opening a source.
        self._stopping.set()
        if self._task is not None:
            await asyncio.shield(self._task)

    async def start(self):
        if not self.settings.workflow_autostart:
            return
        self._status = "starting"

        def start_snapshot():
            if self._stopping.is_set():
                return None
            if not self.settings.workflow_id:
                raise EdgeError(
                    "Select a saved local Workflow before enabling automatic startup",
                    code="workflow_not_selected",
                    status_code=400,
                )
            specification = self.definitions.resolve(
                "local", self.settings.workflow_id, use_cache=False
            )
            specification = self.video.workflows.resolve_definition(specification)
            if self._stopping.is_set():
                return None
            image_input = select_single_image_input(specification)
            reference = "kit://camera"
            if self.settings.video_source == "rtsp":
                reference = self.settings.rtsp_url.strip()
                try:
                    parsed = urlsplit(reference)
                    valid = parsed.scheme in ("rtsp", "rtsps") and parsed.hostname
                    parsed.port
                except ValueError:
                    valid = False
                if not valid:
                    raise EdgeError(
                        "Configure a valid RTSP URL before starting the Workflow",
                        code="invalid_rtsp_url", status_code=400,
                    )
            metadata = next((item["name"] for item in specification.get("inputs", [])
                             if item.get("type") == "WorkflowVideoMetadata"), "video_metadata")
            self._digest = hashlib.sha256(
                json.dumps(specification, sort_keys=True).encode()
            ).hexdigest()
            # Own this run's specification and parameters. Saving a draft while
            # the pipeline runs does not change its deployed graph or inputs.
            return self.video.start(
                specification=copy.deepcopy(specification),
                image_input=image_input,
                inputs=copy.deepcopy(self.settings.workflow_parameters),
                video_reference=reference,
                max_fps=min(self.settings.workflow_fps, self.settings.max_fps),
                video_metadata_input_name=metadata,
            )

        try:
            result = await self.control(start_snapshot)
            if result is None:
                self._status = "stopped"
                return
            self._pipeline_id = result["pipeline_id"]
            self._status = result["status"]
        except Exception as exc:
            self._status = "failed"
            self._error = {
                "code": getattr(exc, "code", "workflow_startup_failed"),
                "message": video_error(exc),
            }

    def status(self):
        pipeline = self.video.status() if self._pipeline_id else None
        current = pipeline is not None and pipeline["pipeline_id"] == self._pipeline_id
        status = pipeline["status"] if current else self._status
        if self._pipeline_id and not current:
            status = "stopped"
        return {
            "workflow_id": self.settings.workflow_id,
            "autostart": self.settings.workflow_autostart,
            "video_source": self.settings.video_source,
            "target_fps": min(self.settings.workflow_fps, self.settings.max_fps),
            "status": status,
            "pipeline_id": self._pipeline_id,
            "specification_sha256": self._digest,
            "error": (
                {"code": "workflow_execution_failed", "message": pipeline["error"]}
                if current and pipeline.get("error")
                else self._error
            ),
        }
