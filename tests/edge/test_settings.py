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


def test_real_appmgr_config_is_read_by_kit_and_survives_app_directory_change(tmp_path):
    sdk = Path(
        os.getenv(
            "RECAMERA_SDK_ROOT",
            str(ROOT.parent / "RV1126B_Linux_IPC_SDK/project/app/recamera-pro-ext-api"),
        )
    )
    if not (sdk / "kit/config.py").is_file():
        pytest.skip("Set RECAMERA_SDK_ROOT to test the real SDK config contract")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(sdk / "market"), str(sdk), str(ROOT))),
        APPMGR_APPS_DIR=str(tmp_path / "apps"),
        APPMGR_APPDATA_DIR=str(tmp_path / "appdata"),
        INFERENCE_RUNTIME_PROFILE="rv1126b",
    )
    script = r"""
import json
from pathlib import Path
from appmgr import config, manifest, paths
from kit import config as kit_config
from inference.edge.settings import EdgeSettings

template = json.loads(Path(TEMPLATE).read_text())
manifest.validate_manifest(template, allow_v1=False)
items = config.schema_specs(template)
assert items["api_token"]["type"] == "password"
assert all(item["apply"] == "restart" for item in items.values())
defaults = config.schema_defaults(template)
assert defaults["host"] == "127.0.0.1" and defaults["port"] == 9001
clean, errors = config.validate_config(template, {
    "host": "0.0.0.0", "port": 19001, "api_token": "test-secret", "max_fps": 4,
    "workflow_id": "saved-camera", "video_source": "rtsp",
    "rtsp_url": "rtsp://user:private@camera.invalid/live", "workflow_fps": 2,
    "workflow_parameters": '{"confidence": 0.6, "classes": ["person"]}',
})
assert not errors, errors
config.write_user_config(template["id"], clean)
for name in ("old-release", "new-release"):
    app_dir = Path(paths.APPS_DIR) / name
    app_dir.mkdir(parents=True)
    (app_dir / "manifest.json").write_text(json.dumps(template))
    effective = kit_config.effective_config(str(app_dir))
    storage = Path(kit_config.appdata_root()) / template["id"]
    settings = EdgeSettings.from_app_config(effective, model_root=app_dir / "models", storage_root=storage)
    assert settings.host == "0.0.0.0" and settings.port == 19001
    assert settings.api_token == "test-secret" and settings.max_fps == 4
    assert settings.workflow_id == "saved-camera" and settings.workflow_autostart is True
    assert settings.video_source == "rtsp" and settings.workflow_fps == 2
    assert "private@camera" not in repr(settings)
    assert settings.workflow_parameters == {"confidence": 0.6, "classes": ["person"]}
    assert settings.storage_root == Path(paths.appdata_dir(template["id"]))
    assert Path(kit_config.user_config_path(str(app_dir))) == storage / "config.json"
for bad in ({"max_fps": 61}, {"port": False}, {"storage_root": "/system"}):
    _, errors = config.validate_config(template, bad)
    assert errors, bad
print("persisted schema and Kit mapping verified")
"""
    script = (
        "TEMPLATE = "
        + repr(str(ROOT / "deploy/rv1126b/manifest.template.json"))
        + "\n"
        + script
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
