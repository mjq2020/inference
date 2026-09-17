"""Run startup checks in clean processes with heavy imports actively rejected."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
LATCH = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
STARTUP_ERROR = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_STARTUP_ERROR"
HEAVY_IMPORT_GUARD = """
import importlib.abc
import sys

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'torchvision', 'onnxruntime', 'inference_models',
            'transformers', 'diffusers', 'rknn', 'rknnlite',
            'aiohttp', 'redis',
        }:
            self.attempted.append(fullname)
            raise AssertionError('Forbidden startup import: ' + fullname)

sys.meta_path.insert(0, RejectHeavyImports())
"""


def run_python(code, tmp_path, env=None, guard=True):
    child_env = os.environ.copy()
    for name in list(child_env):
        if name in {
            LATCH,
            STARTUP_ERROR,
            "OFFLINE_MODE",
            "INFERENCE_RUNTIME_PROFILE",
            "MAX_ACTIVE_MODELS",
            "VIDEO_SOURCE_BUFFER_SIZE",
            "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE",
            "WORKFLOWS_MAX_CONCURRENT_STEPS",
            "ENABLE_TENSOR_DATA_REPRESENTATION",
            "USE_INFERENCE_MODELS",
            "USE_PYTORCH_FOR_PREPROCESSING",
        }:
            child_env.pop(name)
    child_env.update(
        INFERENCE_RUNTIME_PROFILE="rv1126b",
        DISABLE_VERSION_CHECK="True",
        PYTHONPATH=str(REPOSITORY),
    )
    child_env.update(env or {})
    if guard:
        code = (
            HEAVY_IMPORT_GUARD
            + code
            + "\nassert not RejectHeavyImports.attempted, RejectHeavyImports.attempted\n"
        )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rv1126b_imports_public_core_without_inference_models(tmp_path):
    run_python(
        """
import inference
import inference.models
from inference.core import env
from inference.core.utils.onnx import get_onnxruntime_execution_providers

assert env.INFERENCE_RUNTIME_PROFILE == 'rv1126b'
assert not env.USE_INFERENCE_MODELS
assert not env.USE_PYTORCH_FOR_PREPROCESSING
assert not env.ENABLE_TENSOR_DATA_REPRESENTATION
assert not env.ACTIVE_LEARNING_ENABLED
assert not env.CORE_MODELS_ENABLED
assert not env.ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
assert not env.LOAD_ENTERPRISE_BLOCKS
assert env.WORKFLOWS_IMAGE_TENSOR_DEVICE is None
assert env.MAX_ACTIVE_MODELS == 1
assert env.DEFAULT_BUFFER_SIZE == 2
assert env.PREDICTIONS_QUEUE_SIZE == 2
assert env.WORKFLOWS_MAX_CONCURRENT_STEPS == 1
assert get_onnxruntime_execution_providers('[CPUExecutionProvider]') == ['CPUExecutionProvider']
for operation in [lambda: inference.get_model, lambda: inference.models.YOLOv8ObjectDetection]:
    try:
        operation()
    except RuntimeError as error:
        assert 'rv1126b' in str(error)
    else:
        raise AssertionError('full-runtime model API was exposed on the device')
""",
        tmp_path,
        env={
            "USE_INFERENCE_MODELS": "True",
            "USE_PYTORCH_FOR_PREPROCESSING": "True",
            "ENABLE_TENSOR_DATA_REPRESENTATION": "True",
            "ACTIVE_LEARNING_ENABLED": "True",
            "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
            "LOAD_ENTERPRISE_BLOCKS": "True",
        },
    )


def test_numpy_onnx_session_helper_does_not_import_runtime(tmp_path):
    run_python(
        """
import numpy as np
from inference.core.utils.onnx import run_session_via_iobinding

class Session:
    def run(self, outputs, values):
        assert outputs is None
        return [values['image'] + 1]

image = np.zeros((2, 2), dtype=np.uint8)
actual = run_session_via_iobinding(Session(), 'image', image)
assert len(actual) == 1
np.testing.assert_array_equal(actual[0], np.ones((2, 2), dtype=np.uint8))
""",
        tmp_path,
    )


def test_shared_image_schemas_and_memory_cache_do_not_load_remote_clients(tmp_path):
    run_python(
        """
import numpy as np
from inference.core.cache import cache
from inference.core.utils.image_utils import load_image_from_numpy_object
from inference.core.entities.responses.inference import ObjectDetectionPrediction

image = np.zeros((4, 6, 3), dtype=np.uint8)
np.testing.assert_array_equal(load_image_from_numpy_object(image), image)
with cache.lock('edge-test-lock'):
    cache.set('example', 42)
assert cache.get('example') == 42
prediction = ObjectDetectionPrediction(
    x=2, y=2, width=2, height=2, confidence=0.9, class_id=0, **{'class': 'person'}
)
assert prediction.model_dump(by_alias=True)['class'] == 'person'
assert 'inference_sdk.http.client' not in sys.modules
assert 'inference.core.entities.requests.sam3' not in sys.modules
assert 'inference.core.entities.responses.sam3' not in sys.modules
""",
        tmp_path,
    )


def test_edge_usage_decorator_preserves_sync_async_calls_and_errors(tmp_path):
    run_python(
        """
import asyncio
from inference.usage_tracking.edge import usage_collector

@usage_collector('workflows')
def sync_call(value):
    return value + 1

@usage_collector('workflow_block')
async def async_call(value):
    return value + 2

@usage_collector('workflow_block')
def failing_call():
    raise LookupError('preserve the original exception')

assert sync_call(1, usage_fps=30, usage_api_key='unused') == 2
assert asyncio.run(async_call(1, usage_billable=False)) == 3
try:
    failing_call(usage_workflow_preview=True)
except LookupError as error:
    assert str(error) == 'preserve the original exception'
else:
    raise AssertionError('decorator swallowed the exception')
assert 'inference.usage_tracking.collector' not in sys.modules
""",
        tmp_path,
    )


@pytest.mark.parametrize("offline", ["True", "False"])
def test_offline_latch_survives_reload_and_child_process(tmp_path, offline):
    run_python(
        """
import importlib
import os
import subprocess
import warnings
from inference import _edge_bootstrap as bootstrap
from inference.core import env

expected = os.environ['OFFLINE_MODE'] == 'True'
assert bootstrap.OFFLINE_MODE is expected
assert env.OFFLINE_MODE is expected
os.environ['OFFLINE_MODE'] = str(not expected)
importlib.reload(bootstrap)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    importlib.reload(env)
assert env.OFFLINE_MODE is expected
assert any('Changing OFFLINE_MODE' in str(w.message) for w in captured)
code = 'from inference._edge_bootstrap import OFFLINE_MODE; assert OFFLINE_MODE is ' + str(expected)
subprocess.run([sys.executable, '-c', code], env=os.environ, check=True)
if expected:
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
    assert os.environ['YOLO_OFFLINE'] == 'True'
""",
        tmp_path,
        env={"OFFLINE_MODE": offline},
    )


def test_dotenv_can_select_profile_and_offline_mode(tmp_path):
    (tmp_path / ".env").write_text(
        "INFERENCE_RUNTIME_PROFILE=rv1126b\nOFFLINE_MODE=True\n"
    )
    run_python(
        """
import os
os.environ.pop('INFERENCE_RUNTIME_PROFILE')
from inference.runtime import IS_RV1126B
from inference._edge_bootstrap import OFFLINE_MODE
assert IS_RV1126B and OFFLINE_MODE
assert os.environ['INFERENCE_RUNTIME_PROFILE'] == 'rv1126b'
""",
        tmp_path,
    )


@pytest.mark.parametrize("contents", ["OFFLINE_MODE\n", "OFFLINE_MODE=invalid\n"])
def test_invalid_offline_dotenv_remains_failed_after_environment_mutation(
    tmp_path, contents
):
    (tmp_path / ".env").write_text(contents)
    run_python(
        """
import os
for attempt in range(2):
    try:
        import inference
    except ValueError as error:
        assert 'OFFLINE_MODE' in str(error)
    else:
        raise AssertionError('malformed startup configuration was accepted')
    os.environ['OFFLINE_MODE'] = 'False'
""",
        tmp_path,
    )


def test_unknown_runtime_profile_is_rejected_before_heavy_imports(tmp_path):
    run_python(
        """
try:
    import inference
except ValueError as error:
    assert 'INFERENCE_RUNTIME_PROFILE' in str(error)
else:
    raise AssertionError('unknown runtime profile was accepted')
""",
        tmp_path,
        env={"INFERENCE_RUNTIME_PROFILE": "rv1126"},
    )


def test_full_profile_keeps_model_bootstrap_and_existing_defaults(tmp_path):
    # Stand in only for the separately packaged full model distribution. This
    # verifies dispatch/default compatibility without installing Torch in CI.
    run_python(
        """
import importlib.abc
import importlib.util
import os
import sys

class ModelPackage(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    loaded = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'inference_models', 'inference_models.configuration'}:
            return importlib.util.spec_from_loader(fullname, self, is_package=fullname == 'inference_models')
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        self.loaded.append(module.__name__)
        module.OFFLINE_MODE = False
        os.environ.setdefault('HF_HUB_CACHE', '/tmp/cache/hf_home/hub')

loader = ModelPackage()
sys.meta_path.insert(0, loader)
import inference
assert loader.loaded == ['inference_models']
from inference.core import env
assert loader.loaded == ['inference_models', 'inference_models.configuration']
assert env.INFERENCE_RUNTIME_PROFILE == 'full'
assert env.MAX_ACTIVE_MODELS == 8
assert env.DEFAULT_BUFFER_SIZE == 64
assert env.PREDICTIONS_QUEUE_SIZE == 512
assert env.WORKFLOWS_MAX_CONCURRENT_STEPS == 8
""",
        tmp_path,
        env={"INFERENCE_RUNTIME_PROFILE": "full"},
        guard=False,
    )
