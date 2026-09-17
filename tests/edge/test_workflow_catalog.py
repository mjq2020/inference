"""Run the restored original catalog without model-framework imports."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = """
import importlib.abc
import sys
class FrameworkGuard(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'torchvision', 'onnxruntime', 'inference_models',
            'transformers', 'diffusers',
        }:
            self.attempted.append(fullname)
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, FrameworkGuard())
import numpy as np
from inference.core.workflows.execution_engine.core import ExecutionEngine
"""


def run_catalog_script(source):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BOOTSTRAP
            + textwrap.dedent(source)
            + "\nassert not FrameworkGuard.attempted, FrameworkGuard.attempted",
        ],
        cwd=ROOT,
        env={**os.environ, "INFERENCE_RUNTIME_PROFILE": "rv1126b"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalog_exposes_original_manifests_and_detection_versions_without_torch():
    run_catalog_script(
        """
        from inference.core.workflows.core_steps.catalog_rv1126b import load_numpy_blocks
        from inference.core.workflows.core_steps.models.rknn.v1 import (
            BlockManifest, BlockManifestV2, BlockManifestV3,
            OriginalManifestV1, OriginalManifestV2, OriginalManifestV3,
        )
        from inference.core.workflows.execution_engine.introspection.blocks_loader import describe_available_blocks
        blocks = load_numpy_blocks()
        assert len(blocks) >= 100
        identifiers = [identifier for block in blocks for identifier in block.get_manifest().model_fields['type'].annotation.__args__]
        assert len(identifiers) == len(set(identifiers))
        assert 'roboflow_core/byte_tracker@v3' in identifiers
        assert 'roboflow_core/label_visualization@v2' in identifiers
        assert 'roboflow_core/sam@v1' not in identifiers
        description = describe_available_blocks(dynamic_blocks=[])
        assert len(description.blocks) == len(blocks)
        for original, adapted in [(OriginalManifestV1, BlockManifest), (OriginalManifestV2, BlockManifestV2), (OriginalManifestV3, BlockManifestV3)]:
            assert original.describe_outputs() == adapted.describe_outputs()
            for name, field in original.model_fields.items():
                if name != 'type':
                    assert adapted.model_fields[name].default == field.default, name
                    assert adapted.model_fields[name].annotation == field.annotation, name
            manifest = adapted(type=adapted.model_fields['type'].annotation.__args__[0], name='detect', images='$inputs.image', model_id='detector/1')
            assert manifest.max_detections == 300
            assert manifest.discover_dependent_resources() == []
        """
    )


def test_saved_inner_workflow_resolution_bounds_cycles_without_rejecting_reuse():
    run_catalog_script(
        """
        import copy
        from collections import Counter
        from inference.core.workflows.execution_engine.v1.inner_workflow import reference_resolution as refs
        from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
            InnerWorkflowCompositionCycleError,
            InnerWorkflowNestingDepthError,
            InnerWorkflowTotalCountError,
        )
        def step(name, reference):
            return dict(type='roboflow_core/inner_workflow@v1', name=name,
                        workflow_workspace_id='local', workflow_id=reference)
        def workflow(*steps):
            return dict(version='1.0', inputs=[], steps=list(steps), outputs=[])
        saved = {}
        calls = Counter()
        def resolver(workspace, name, version, parameters):
            calls[name] += 1
            return saved[name]
        parameters = {refs.WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver}
        for documents in (
            {'a': workflow(step('self', 'a'))},
            {'a': workflow(step('ab', 'b')), 'b': workflow(step('ba', 'a'))},
        ):
            saved.clear(); saved.update(documents); calls.clear()
            before = copy.deepcopy(saved)
            try:
                refs.normalize_inner_workflow_references_in_definition(workflow(step('root', 'a')), parameters)
            except InnerWorkflowCompositionCycleError as error:
                assert 'cycle' in str(error).lower()
            else:
                raise AssertionError('reference cycle was accepted')
            assert saved == before
            assert all(count == 1 for count in calls.values())
        saved.clear(); saved['leaf'] = workflow(); calls.clear()
        source = workflow(step('first', 'leaf'), step('second', 'leaf'))
        result = refs.normalize_inner_workflow_references_in_definition(source, parameters)
        assert calls == {'leaf': 1}
        first, second = [s['workflow_definition'] for s in result['steps']]
        assert first == second == saved['leaf']
        assert first is not second and first is not saved['leaf']
        assert 'workflow_id' in source['steps'][0]
        for index in range(refs.WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH + 1):
            saved[str(index)] = workflow(step('deeper', str(index + 1)))
        try:
            refs.normalize_inner_workflow_references_in_definition(workflow(step('root', '0')), parameters)
        except InnerWorkflowNestingDepthError:
            pass
        else:
            raise AssertionError('depth limit was not checked during expansion')
        try:
            refs.normalize_inner_workflow_references_in_definition(
                workflow(*(step(str(i), 'leaf') for i in range(refs.WORKFLOWS_MAX_INNER_WORKFLOW_COUNT + 1))), parameters)
        except InnerWorkflowTotalCountError:
            pass
        else:
            raise AssertionError('count limit was not checked during expansion')
        """
    )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_standard_detection_defaults_filters_and_outputs(version):
    run_catalog_script(
        f"version = {version}\n"
        + """
class Manager:
    def __init__(self):
        self.requests = []
    def infer_from_request_sync(self, model_id, request):
        self.requests.append(request)
        assert request['image']['value'].shape == (64, 64, 3)
        return {'inference_id': 'test', 'image': {'width': 64, 'height': 64},
                'predictions': [{'x': 20., 'y': 20., 'width': 4., 'height': 4.,
                                 'confidence': .9, 'class': 'person' if n % 2 else 'car',
                                 'class_id': n % 2} for n in range(40)]}
manager = Manager()
step = {'type': f'roboflow_core/roboflow_object_detection_model@v{version}',
        'name': 'detect', 'images': '$inputs.image', 'model_id': 'detector/1',
        'class_filter': [], 'active_learning_target_dataset': 'ignored-when-disabled'}
spec = {'version': '1.0', 'inputs': [{'type': 'WorkflowImage', 'name': 'image'}],
        'steps': [step], 'outputs': [{'type': 'JsonField', 'name': 'predictions', 'selector': '$steps.detect.predictions'}]}
if version >= 2:
    spec['outputs'].append({'type': 'JsonField', 'name': 'model', 'selector': '$steps.detect.model_id'})
def run():
    engine = ExecutionEngine.init(workflow_definition=spec, init_parameters={'model_manager': manager})
    return engine.run(runtime_parameters={'image': np.zeros((64, 64, 3), np.uint8)})[0]
result = run()
assert len(result['predictions']) == 40
assert manager.requests[-1]['confidence'] == .4
assert manager.requests[-1]['iou_threshold'] == .3
assert manager.requests[-1]['max_detections'] == 300
assert manager.requests[-1]['class_filter'] is None
if version >= 2:
    assert result['model'] == 'detector/1'
step['class_filter'] = ['person']
result = run()
assert len(result['predictions']) == 20
assert set(result['predictions']['class_name']) == {'person'}
assert manager.requests[-1]['class_filter'] is None
if version == 3:
    step.update(confidence_mode='custom', custom_confidence=.61)
    run()
    assert manager.requests[-1]['confidence'] == .61
    step['confidence_mode'] = 'default'
    run()
    assert manager.requests[-1]['confidence'] == .4
spec['steps'].insert(0, {'type': 'roboflow_core/convert_grayscale@v1',
                       'name': 'gray', 'image': '$inputs.image'})
step['images'] = '$steps.gray.image'
assert len(run()['predictions']) == 20
"""
    )


def test_original_opencv_graph_and_focus_v2_match_numpy_pixels():
    run_catalog_script(
        """
        import cv2
        image = np.random.default_rng(8).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        spec = {'version': '1.0', 'inputs': [{'type': 'WorkflowImage', 'name': 'image'}],
                'steps': [
                    {'type': 'roboflow_core/convert_grayscale@v1', 'name': 'gray', 'image': '$inputs.image'},
                    {'type': 'roboflow_core/image_blur@v1', 'name': 'blur', 'image': '$steps.gray.image'},
                    {'type': 'roboflow_core/threshold@v1', 'name': 'threshold', 'image': '$steps.blur.image', 'thresh_value': 120},
                    {'type': 'roboflow_core/camera_focus@v2', 'name': 'focus', 'image': '$inputs.image',
                     'show_zebra_warnings': False, 'grid_overlay': 'None', 'show_hud': False,
                     'show_focus_peaking': False, 'show_center_marker': False},
                ], 'outputs': [
                    {'type': 'JsonField', 'name': 'image', 'selector': '$steps.threshold.image'},
                    {'type': 'JsonField', 'name': 'focus', 'selector': '$steps.focus.focus_measure'},
                ]}
        engine = ExecutionEngine.init(workflow_definition=spec)
        result = engine.run(runtime_parameters={'image': image})[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 120, 255, cv2.THRESH_BINARY)[1]
        np.testing.assert_array_equal(result['image'].numpy_image, expected)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        np.testing.assert_allclose(result['focus'], np.mean(dx * dx + dy * dy))
        assert result['image']._tensor_image is None
        """
    )


def test_tracking_counting_and_visualization_share_original_video_state():
    run_catalog_script(
        """
        from datetime import datetime, timezone, timedelta
        class Manager:
            y = 26
            def infer_from_request_sync(self, model_id, request):
                result = {'inference_id': 'test', 'image': {'width': 64, 'height': 64},
                          'predictions': [{'x': 30., 'y': float(self.y), 'width': 20., 'height': 20.,
                                           'confidence': .95, 'class': 'person', 'class_id': 0}]}
                self.y += 4
                return result
        manager = Manager()
        spec = {'version': '1.0', 'inputs': [{'type': 'WorkflowImage', 'name': 'image'}],
                'steps': [
                    {'type': 'ObjectDetectionModel', 'name': 'detect', 'images': '$inputs.image', 'model_id': 'detector/1'},
                    {'type': 'roboflow_core/byte_tracker@v2', 'name': 'track', 'image': '$inputs.image', 'detections': '$steps.detect.predictions'},
                    {'type': 'roboflow_core/line_counter@v2', 'name': 'line', 'image': '$inputs.image', 'detections': '$steps.track.tracked_detections', 'line_segment': [[0, 32], [64, 32]], 'triggering_anchor': 'CENTER'},
                    {'type': 'BoundingBoxVisualization', 'name': 'boxes', 'image': '$inputs.image', 'predictions': '$steps.track.tracked_detections'},
                    {'type': 'roboflow_core/label_visualization@v2', 'name': 'labels', 'image': '$steps.boxes.image', 'predictions': '$steps.track.tracked_detections'},
                ], 'outputs': [
                    {'type': 'JsonField', 'name': 'tracks', 'selector': '$steps.track.tracked_detections'},
                    {'type': 'JsonField', 'name': 'in', 'selector': '$steps.line.count_in'},
                    {'type': 'JsonField', 'name': 'out', 'selector': '$steps.line.count_out'},
                    {'type': 'JsonField', 'name': 'image', 'selector': '$steps.labels.image'},
                ]}
        engine = ExecutionEngine.init(workflow_definition=spec, init_parameters={'model_manager': manager})
        now = datetime.now(timezone.utc)
        ids = []
        source = np.zeros((64, 64, 3), np.uint8)
        for frame in range(4):
            image = {'type': 'numpy', 'value': source,
                     'video_metadata': {'video_identifier': 'camera', 'frame_number': frame + 1,
                         'frame_timestamp': now + timedelta(seconds=frame / 10), 'fps': 10, 'comes_from_video_file': False}}
            result = engine.run(runtime_parameters={'image': image}, fps=10)[0]
            ids.append(result['tracks'].tracker_id.tolist())
            assert result['image'].numpy_image.shape == source.shape
            assert np.count_nonzero(result['image'].numpy_image) > 0
        assert ids == [[1]] * 4, ids
        assert result['in'] + result['out'] == 1, result
        assert not source.any(), 'copy_image must preserve the original input'
        """
    )


def test_rate_limiter_uses_live_timestamps_and_file_frame_time():
    run_catalog_script(
        """
        from datetime import datetime, timezone, timedelta
        from types import SimpleNamespace
        from inference.core.workflows.core_steps.flow_control.rate_limiter.v1 import RateLimiterBlockV1
        for from_file in (False, True):
            limiter = RateLimiterBlockV1()
            now = datetime.now(timezone.utc)
            def run(number, seconds):
                metadata = SimpleNamespace(comes_from_video_file=from_file, fps=10,
                    frame_number=number, frame_timestamp=now + timedelta(seconds=seconds))
                return limiter.run(cooldown_seconds=2, depends_on=None,
                    next_steps=['$steps.next'], video_reference_image=SimpleNamespace(video_metadata=metadata))
            assert run(1, 0).context == ['$steps.next']
            # Five seconds elapsed in a live stream even though frames were
            # dropped. File playback still advances by just one tenth second.
            assert run(2, 5).context == (None if from_file else ['$steps.next'])
        """
    )
