"""Small, explicit resource budgets for the device runtime."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class EdgeSettings:
    model_root: Path = Path("models")
    storage_root: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 9001
    api_token: str = field(default="", repr=False)
    roboflow_api_key: str = field(default="", repr=False)
    workflow_id: str = ""
    workflow_autostart: bool = False
    workflow_parameters: dict = field(default_factory=dict, repr=False)
    video_source: str = "camera"
    rtsp_url: str = field(default="", repr=False)
    workflow_fps: float = 3.0
    max_request_bytes: int = 32 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 4096 * 2160
    max_video_output_pixels: int = 1280 * 720
    max_pending_requests: int = 2
    workflow_cache_size: int = 4
    max_workflow_steps: int = 128
    max_fps: float = 10.0

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty address")
        if not isinstance(self.api_token, str):
            raise ValueError("api_token must be a string")
        if not isinstance(self.roboflow_api_key, str):
            raise ValueError("roboflow_api_key must be a string")
        if not isinstance(self.workflow_id, str) or (
            self.workflow_id and not re.fullmatch(r"[\w-]{1,128}", self.workflow_id)
        ):
            raise ValueError(
                "workflow_id must be an empty string or a local Workflow identifier"
            )
        if type(self.workflow_autostart) is not bool:
            raise ValueError("workflow_autostart must be a boolean")
        if not isinstance(self.workflow_parameters, dict):
            raise ValueError("workflow_parameters must be a JSON object")
        if self.video_source not in ("camera", "rtsp"):
            raise ValueError("video_source must be camera or rtsp")
        if not isinstance(self.rtsp_url, str):
            raise ValueError("rtsp_url must be a string")
        if type(self.workflow_fps) not in (int, float) or not 0 < self.workflow_fps <= 60:
            raise ValueError("workflow_fps must be between 0 and 60")
        try:
            json.dumps(self.workflow_parameters, allow_nan=False)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError(
                "workflow_parameters must contain valid JSON values"
            ) from exc
        if self.host not in ("127.0.0.1", "::1", "localhost") and not self.api_token:
            raise ValueError(
                "A non-loopback listener requires INFERENCE_EDGE_API_TOKEN"
            )
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        for name in (
            "max_request_bytes",
            "max_response_bytes",
            "max_image_pixels",
            "max_video_output_pixels",
            "max_pending_requests",
            "workflow_cache_size",
            "max_workflow_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_fps, bool)
            or not isinstance(self.max_fps, (int, float))
            or not 0 < self.max_fps <= 60
        ):
            raise ValueError("max_fps must be between 0 and 60")

    @classmethod
    def from_env(cls, **overrides):
        bindings = {
            "model_root": ("MODEL_ROOT", "models", Path),
            "storage_root": ("STORAGE_ROOT", "data", Path),
            "host": ("HOST", "127.0.0.1", str),
            "port": ("PORT", "9001", int),
            "api_token": ("API_TOKEN", "", str),
            "roboflow_api_key": ("ROBOFLOW_API_KEY", "", str),
            "workflow_id": ("WORKFLOW_ID", "", str),
            "workflow_autostart": ("WORKFLOW_AUTOSTART", "false", cls._boolean),
            "workflow_parameters": ("WORKFLOW_PARAMETERS", "{}", cls._parameters),
            "video_source": ("VIDEO_SOURCE", "camera", str),
            "rtsp_url": ("RTSP_URL", "", str),
            "workflow_fps": ("WORKFLOW_FPS", "3", float),
            "max_request_bytes": ("MAX_REQUEST_BYTES", "33554432", int),
            "max_response_bytes": ("MAX_RESPONSE_BYTES", "8388608", int),
            "max_image_pixels": ("MAX_IMAGE_PIXELS", "8847360", int),
            "max_video_output_pixels": ("MAX_VIDEO_OUTPUT_PIXELS", "921600", int),
            "max_fps": ("MAX_FPS", "10", float),
        }
        values = {
            name: (
                overrides[name]
                if name in overrides
                else convert(os.getenv("INFERENCE_EDGE_" + suffix, default))
            )
            for name, (suffix, default, convert) in bindings.items()
        }
        values.update(overrides)
        return cls(**values)

    @staticmethod
    def _boolean(value):
        if isinstance(value, str) and value.lower() in ("true", "1", "false", "0"):
            return value.lower() in ("true", "1")
        raise ValueError("WORKFLOW_AUTOSTART must be true or false")

    @staticmethod
    def _parameters(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, RecursionError) as exc:
                raise ValueError("workflow_parameters must be a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("workflow_parameters must be a JSON object")
        return value

    @classmethod
    def from_app_config(cls, config: Mapping, *, model_root: Path, storage_root: Path):
        """Apply Kit's persisted configuration without moving managed assets.

        Kit already overlays manifest defaults with appdata/config.json. These
        values win over development environment variables. File destinations
        are supplied by the installed entry, never by user-editable parameters.
        """
        if not isinstance(config, Mapping):
            raise ValueError("App configuration must be an object")
        overrides = {"model_root": model_root, "storage_root": storage_root}
        if "host" in config and config["host"] not in ("127.0.0.1", "0.0.0.0"):
            raise ValueError("host must be 127.0.0.1 or 0.0.0.0")
        for name in (
            "host",
            "api_token",
            "roboflow_api_key",
            "workflow_id",
            "workflow_autostart",
            "video_source",
            "rtsp_url",
        ):
            if name in config:
                overrides[name] = config[name]
        # App Center starts the selected workload; an empty selection starts
        # the editor service. The old independent autostart toggle is retired.
        overrides["workflow_autostart"] = bool(overrides.get("workflow_id", ""))
        if "workflow_parameters" in config:
            overrides["workflow_parameters"] = cls._parameters(
                config["workflow_parameters"]
            )
        bounds = {
            "workflow_fps": (0.1, 60, "workflow_fps", 1, False),
            "port": (1024, 65535, "port", 1, True),
            "max_fps": (0.1, 60, "max_fps", 1, False),
            "max_workflow_steps": (1, 256, "max_workflow_steps", 1, True),
            "max_request_mib": (1, 64, "max_request_bytes", 1024 * 1024, True),
            "max_response_kib": (64, 16384, "max_response_bytes", 1024, True),
            "max_image_pixels": (65536, 4096 * 2160, "max_image_pixels", 1, True),
            "max_video_output_pixels": (
                307200,
                2073600,
                "max_video_output_pixels",
                1,
                True,
            ),
        }
        for key, (minimum, maximum, name, factor, integer) in bounds.items():
            if key not in config:
                continue
            value = config[key]
            valid_type = type(value) is int if integer else type(value) in (int, float)
            if not valid_type or not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            overrides[name] = value * factor
        return cls.from_env(**overrides)
