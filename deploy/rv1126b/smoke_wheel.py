#!/usr/bin/env python3
"""Install only the source/derived wheels into a temporary, isolated path."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("--unwritable-home", action="store_true")
    arguments = parser.parse_args()
    wheels = [
        next(arguments.wheelhouse.glob("inference_rv1126b-*.whl")),
        next(arguments.wheelhouse.glob("supervision_rv1126b-*.whl")),
        next(arguments.wheelhouse.glob("networkx_rv1126b-*.whl")),
    ]
    with tempfile.TemporaryDirectory(prefix="inference-wheel-smoke-") as temporary:
        target = Path(temporary) / "site-packages"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(target),
                *map(str, wheels),
            ],
            check=True,
        )
        code = r"""
import os
import sys
import importlib.abc
from pathlib import Path

class FirmwareWithoutCompression(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'bz2', '_bz2', 'lzma', '_lzma'}:
            raise ModuleNotFoundError('Simulated trimmed firmware module: ' + fullname, name=fullname)

for name in ('bz2', '_bz2', 'lzma', '_lzma'):
    sys.modules.pop(name, None)
sys.meta_path.insert(0, FirmwareWithoutCompression())
import inference_edge_probe
import numpy as np
import networkx as nx
from inference.edge.api import create_app
from inference.edge.settings import EdgeSettings
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.detection.geometry import is_point_in_zone

assert inference_edge_probe.SUPPORTED_BLOCK_COUNT >= 10
application = create_app(EdgeSettings(model_root=Path.cwd() / 'models'))
assert application.openapi()['info']['title'] == 'Inference for RV1126B'
workflow = {
    'version': '1.0',
    'inputs': [{'type': 'WorkflowImage', 'name': 'image'}],
    'steps': [{'type': 'AbsoluteStaticCrop', 'name': 'crop', 'image': '$inputs.image',
               'x_center': 6, 'y_center': 5, 'width': 4, 'height': 2}],
    'outputs': [{'type': 'JsonField', 'name': 'crop', 'selector': '$steps.crop.crops'}],
}
image = np.arange(360, dtype=np.uint8).reshape(10, 12, 3)
result = ExecutionEngine.init(workflow_definition=workflow).run(runtime_parameters={'image': image})
np.testing.assert_array_equal(result[0]['crop'].numpy_image, image[4:6, 4:8])
assert is_point_in_zone((2, 2), [(0, 0), (4, 0), (4, 4), (0, 4)])
graph = nx.path_graph(3)
nx.write_adjlist(graph, 'plain.adjlist')
assert nx.is_isomorphic(graph, nx.read_adjlist('plain.adjlist', nodetype=int))
try:
    nx.write_adjlist(graph, 'compressed.adjlist.bz2')
except RuntimeError as error:
    assert 'no _bz2' in str(error)
else:
    raise AssertionError('Unavailable bzip2 graph IO was accepted')
assert not Path('compressed.adjlist.bz2').exists()
target = Path(os.environ['PYTHONPATH']).resolve()
for name, module in sys.modules.copy().items():
    if name.split('.')[0] in {'inference', 'inference_sdk', 'supervision', 'networkx'}:
        source = getattr(module, '__file__', None)
        if source:
            assert Path(source).resolve().is_relative_to(target), (name, source)
    assert name.split('.')[0] not in inference_edge_probe.FORBIDDEN_MODULES, name
print('Isolated wheel API, OpenAPI, actual Workflow crop and Shapely geometry: OK')
"""
        probe_env = {
            **os.environ,
            "PYTHONPATH": str(target),
            "INFERENCE_RUNTIME_PROFILE": "rv1126b",
        }
        if arguments.unwritable_home:
            probe_env["HOME"] = "/nonexistent"
            for key in ("MPLCONFIGDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
                probe_env.pop(key, None)
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=temporary,
            env=probe_env,
            check=True,
        )


if __name__ == "__main__":
    main()
