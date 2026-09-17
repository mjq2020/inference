"""Dependency-light startup for installations without inference-models.

The full distribution retains inference_models._offline as its startup owner.
The RKNN distribution does not ship that package. Both use the same process
latch, including the failed-start marker, so child processes and reloads retain
the original offline decision. No model conversion or inference library is
imported here.
"""

import os
import sys
import tempfile
import warnings

from dotenv import dotenv_values, load_dotenv

OFFLINE_MODE_PROCESS_LATCH_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
_STARTUP_ERROR_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_STARTUP_ERROR"


def _parse_offline_mode(value: object, variable_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Expected {variable_name} to be true or false, got {value!r}.")


def _decide_offline_mode() -> bool:
    inherited = os.environ.get(OFFLINE_MODE_PROCESS_LATCH_ENV)
    if inherited is not None:
        return _parse_offline_mode(inherited, OFFLINE_MODE_PROCESS_LATCH_ENV)
    if "OFFLINE_MODE" in os.environ:
        return _parse_offline_mode(os.environ["OFFLINE_MODE"], "OFFLINE_MODE")
    values = dotenv_values(_DOTENV_PATH)
    if "OFFLINE_MODE" in values:
        return _parse_offline_mode(values["OFFLINE_MODE"], "OFFLINE_MODE")
    return False


def warn_if_offline_mode_changed() -> None:
    try:
        requested = _parse_offline_mode(
            os.getenv("OFFLINE_MODE", "False"), "OFFLINE_MODE"
        )
    except ValueError:
        requested = None
    if requested is None or requested != OFFLINE_MODE:
        warnings.warn(
            "Changing OFFLINE_MODE at runtime is not supported. The new value is "
            "being ignored; restart the process to change offline mode.",
            RuntimeWarning,
            stacklevel=2,
        )


_DOTENV_PATH = os.path.join(os.getcwd(), ".env")
if _STARTUP_ERROR_ENV in os.environ:
    raise ValueError(os.environ[_STARTUP_ERROR_ENV])
_first_establishment = OFFLINE_MODE_PROCESS_LATCH_ENV not in os.environ
try:
    OFFLINE_MODE = _decide_offline_mode()
except ValueError as error:
    os.environ[_STARTUP_ERROR_ENV] = str(error)
    raise

load_dotenv(_DOTENV_PATH)
_cache_root = (
    os.getenv("INFERENCE_HOME") or os.getenv("MODEL_CACHE_DIR") or "/tmp/cache"
)
os.environ.setdefault("HF_HOME", os.path.join(_cache_root, "hf_home"))
os.environ.setdefault(
    "HF_HUB_CACHE",
    os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.getenv("TRANSFORMERS_CACHE")
    or os.path.join(os.environ["HF_HOME"], "hub"),
)
os.environ.setdefault(
    "HF_MODULES_CACHE", os.path.join(os.environ["HF_HOME"], "modules")
)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(tempfile.gettempdir(), "roboflow-inference", "ultralytics"),
)
os.environ[OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)
if OFFLINE_MODE:
    if _first_establishment and any(
        name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules
    ):
        raise RuntimeError(
            "Ultralytics was imported before inference could establish OFFLINE_MODE. "
            "Restart the process and import inference first."
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "True"
