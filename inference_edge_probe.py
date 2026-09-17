"""Application-manager import probe; selects the profile before package import."""

import importlib.abc
import os
import sys

FORBIDDEN_MODULES = {
    "torch",
    "torchvision",
    "onnxruntime",
    "inference_models",
    "transformers",
    "diffusers",
    "tensorflow",
}


class _RejectHeavyImports(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN_MODULES:
            self.attempted.append(fullname)
            raise ImportError(f"Unsupported device runtime dependency: {fullname}")


if any(name.split(".")[0] in FORBIDDEN_MODULES for name in sys.modules):
    raise RuntimeError(
        "A full inference model framework was loaded before the device probe"
    )
os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
_guard = _RejectHeavyImports()
sys.meta_path.insert(0, _guard)
try:
    import cv2
    import numpy

    from inference.core.workflows.core_steps.loader_rv1126b import load_blocks
    from inference.core.workflows.execution_engine.core import ExecutionEngine
    from inference.edge.api import create_app

    if numpy.__version__ != "1.23.5" or cv2.__version__ != "4.6.0":
        raise RuntimeError("Expected firmware NumPy 1.23.5 and OpenCV 4.6.0")
    SUPPORTED_BLOCK_COUNT = len(load_blocks())
    if _guard.attempted:
        raise RuntimeError(
            f"Unexpected model framework import attempts: {_guard.attempted}"
        )
finally:
    sys.meta_path.remove(_guard)
