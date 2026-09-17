#!/usr/bin/env python3
"""Measure the real HTTP/NumPy runtime without loading an NPU model."""

import argparse
import base64
import gc
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/inference-rv1126b-mpl"

FORBIDDEN = {
    "torch",
    "torchvision",
    "onnxruntime",
    "onnx",
    "inference_models",
    "transformers",
    "diffusers",
    "rknnlite",
}


class Guard(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            self.attempted.append(fullname)
            raise ImportError("Device runtime tried to import " + fullname)


sys.meta_path.insert(0, Guard())


def memory():
    values = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {
            "Rss",
            "Pss",
            "Private_Clean",
            "Private_Dirty",
            "Shared_Clean",
            "Shared_Dirty",
        }:
            values[key + "_MiB"] = round(int(value.split()[0]) / 1024, 2)
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            values["peak_rss_MiB"] = round(int(line.split()[1]) / 1024, 2)
    values["threads"] = len(list(Path("/proc/self/task").iterdir()))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--kit-app", type=Path, help="Also include the real Kit application lifecycle"
    )
    args = parser.parse_args()
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "platform": os.uname().machine,
        "python": sys.version.split()[0],
        "scope": "HTTP server + original NumPy Workflow; no model, no NPU, no camera",
        "baseline": memory(),
    }
    try:
        distribution = importlib.metadata.distribution("inference-rv1126b")
        manifest = distribution.read_text("SOURCE_MANIFEST.json")
        report["source_manifest_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
        report["package_version"] = distribution.version
    except importlib.metadata.PackageNotFoundError:
        report["source_manifest_sha256"] = None
    started = time.monotonic()
    import cv2
    import numpy as np
    import uvicorn

    from inference.edge.api import create_app

    cv2.setNumThreads(1)
    kit_app = None
    if args.kit_app:
        report["scope"] = (
            "Kit application + HTTP + original NumPy Workflow; no model, no NPU, no camera"
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            os.environ["INFERENCE_EDGE_PORT"] = str(listener.getsockname()[1])
        spec = importlib.util.spec_from_file_location(
            "edge_app_memory_probe", args.kit_app.resolve()
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kit_app = module.InferenceEdgeApp()
        kit_app.start(
            app_dir=str(args.kit_app.resolve().parent),
            manifest={},
            config={},
            verbose=False,
        )
        server, thread = kit_app._server, kit_app._http_thread
    else:
        app = create_app()
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=0, access_log=False, log_level="error"
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("HTTP startup failed")
            time.sleep(0.05)
        address = "http://127.0.0.1:" + str(
            server.servers[0].sockets[0].getsockname()[1]
        )

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(
                address + path, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)

        assert request("/healthz")["profile"] == "rv1126b"
        report["api_startup_seconds"] = round(time.monotonic() - started, 3)
        report["versions"] = {"numpy": np.__version__, "opencv": cv2.__version__}
        report["http_idle"] = memory()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        specification = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {
                    "type": "AbsoluteStaticCrop",
                    "name": "crop",
                    "image": "$inputs.image",
                    "x_center": 320,
                    "y_center": 240,
                    "width": 224,
                    "height": 224,
                }
            ],
            "outputs": [
                {"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}
            ],
        }
        body = {
            "specification": specification,
            "inputs": {
                "image": {"type": "base64", "value": base64.b64encode(encoded).decode()}
            },
        }
        response = request("/workflows/run", body)
        crop = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(response["outputs"][0]["crop"]["value"]), np.uint8
            ),
            cv2.IMREAD_COLOR,
        )
        assert crop.shape == (224, 224, 3)
        del crop, response
        gc.collect()
        report["workflow_warm"] = memory()
        for _ in range(args.iterations):
            response = request("/workflows/run", body)
            assert response["outputs"][0]["crop"]["type"] == "base64"
            del response
        gc.collect()
        report["iterations"] = args.iterations
        report["workflow_after_iterations"] = memory()
        from inference.core.workflows.execution_engine.v1.compiler.core import (
            COMPILATION_CACHE,
        )

        report["compiled_graphs"] = len(COMPILATION_CACHE._cache)
        report["forbidden_import_attempts"] = Guard.attempted
        report["forbidden_loaded"] = sorted(FORBIDDEN.intersection(sys.modules))
        assert (
            not report["forbidden_import_attempts"] and not report["forbidden_loaded"]
        )
    finally:
        if kit_app is not None:
            kit_app.finish()
        else:
            server.should_exit = True
            thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("HTTP server did not stop")
    report["shutdown_ok"] = True
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
