"""Persisted App Center settings and the real Kit/appmgr schema contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

from inference.edge.settings import EdgeSettings

ROOT = Path(__file__).resolve().parents[2]


def configured(values, tmp_path):
    return EdgeSettings.from_app_config(
        values, model_root=tmp_path / "models", storage_root=tmp_path / "appdata"
    )


def test_persisted_config_wins_over_env_and_cannot_move_managed_data(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("INFERENCE_EDGE_HOST", "0.0.0.0")
    monkeypatch.setenv("INFERENCE_EDGE_PORT", "invalid-old-environment")
    monkeypatch.setenv("INFERENCE_EDGE_STORAGE_ROOT", "/wrong-storage")
    monkeypatch.setenv("INFERENCE_EDGE_MODEL_ROOT", "/wrong-models")
    settings = configured(
        {
            "host": "127.0.0.1",
            "port": 19001,
            "api_token": "",
            "max_fps": 3.5,
            "max_workflow_steps": 12,
            "max_request_mib": 4,
            "max_response_kib": 256,
            "max_image_pixels": 1920 * 1080,
            "max_video_output_pixels": 1920 * 1080,
            "storage_root": "/also-wrong",
        },
        tmp_path,
    )
    assert settings.host == "127.0.0.1"
    assert settings.port == 19001
    assert settings.max_fps == 3.5
    assert settings.max_workflow_steps == 12
    assert settings.max_request_bytes == 4 * 1024 * 1024
    assert settings.max_response_bytes == 256 * 1024
    assert settings.max_image_pixels == 1920 * 1080
    assert settings.max_video_output_pixels == 1920 * 1080
    assert settings.model_root == tmp_path / "models"
    assert settings.storage_root == tmp_path / "appdata"


def test_lan_access_needs_token_and_token_is_not_in_settings_repr(tmp_path):
    with pytest.raises(ValueError, match="TOKEN"):
        configured({"host": "0.0.0.0", "api_token": ""}, tmp_path)
    token = "test-only-private-token"
    settings = configured({"host": "0.0.0.0", "api_token": token}, tmp_path)
    assert settings.api_token == token
    assert token not in repr(settings)


@pytest.mark.parametrize(
    "values",
    [
        {"host": ""},
        {"port": True},
        {"port": 80},
        {"port": 65536},
        {"api_token": 123},
        {"max_fps": True},
        {"max_fps": float("nan")},
        {"max_fps": 60.1},
        {"max_workflow_steps": 257},
        {"max_request_mib": 8.1},
        {"max_response_kib": 16385},
        {"max_image_pixels": 16_000_000},
        {"max_video_output_pixels": 307199},
        {"max_video_output_pixels": 2073601},
    ],
)
def test_hand_edited_config_cannot_bypass_schema_limits(values, tmp_path):
    with pytest.raises(ValueError):
        configured(values, tmp_path)
