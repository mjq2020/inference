"""Explicit NumPy Workflow catalog for the RV1126B runtime.

This list is also the packaging contract: tools may read BLOCK_MODULES without
loading any block. Original modules and manifests are reused; unsupported model
runtimes and optional dependency sets are never silently imported or substituted.
"""

from importlib import import_module
from inspect import isclass

BLOCK_MODULES = (
    "inference.core.workflows.core_steps.classical_cv.auto_rotate_on_edges.v1",
    "inference.core.workflows.core_steps.classical_cv.background_subtraction.v1",
    "inference.core.workflows.core_steps.classical_cv.camera_focus.v1",
    "inference.core.workflows.core_steps.classical_cv.camera_focus.v2",
    "inference.core.workflows.core_steps.classical_cv.contours.v1",
    "inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1",
    "inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1",
    "inference.core.workflows.core_steps.classical_cv.detections_nearest_neighbor.v1",
    "inference.core.workflows.core_steps.classical_cv.distance_measurement.v1",
    "inference.core.workflows.core_steps.classical_cv.dominant_color.v1",
    "inference.core.workflows.core_steps.classical_cv.image_blur.v1",
    "inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1",
    "inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1",
    "inference.core.workflows.core_steps.classical_cv.mask_edge_snap.v1",
    "inference.core.workflows.core_steps.classical_cv.morphological_transformation.v1",
    "inference.core.workflows.core_steps.classical_cv.morphological_transformation.v2",
    "inference.core.workflows.core_steps.classical_cv.motion_detection.v1",
    "inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1",
    "inference.core.workflows.core_steps.classical_cv.sift.v1",
    "inference.core.workflows.core_steps.classical_cv.sift_comparison.v1",
    "inference.core.workflows.core_steps.classical_cv.sift_comparison.v2",
    "inference.core.workflows.core_steps.classical_cv.size_measurement.v1",
    "inference.core.workflows.core_steps.classical_cv.template_matching.v1",
    "inference.core.workflows.core_steps.classical_cv.threshold.v1",
    "inference.core.workflows.core_steps.formatters.current_time.v1",
    "inference.core.workflows.core_steps.formatters.expression.v1",
    "inference.core.workflows.core_steps.formatters.first_non_empty_or_default.v1",
    "inference.core.workflows.core_steps.formatters.json_parser.v1",
    "inference.core.workflows.core_steps.formatters.property_definition.v1",
    "inference.core.workflows.core_steps.formatters.string_template.v1",
    "inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1",
    "inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2",
    "inference.core.workflows.core_steps.formatters.vlm_as_detector.v1",
    "inference.core.workflows.core_steps.formatters.vlm_as_detector.v2",
    "inference.core.workflows.core_steps.transformations.absolute_static_crop.v1",
    "inference.core.workflows.core_steps.transformations.bounding_rect.v1",
    "inference.core.workflows.core_steps.transformations.byte_tracker.v1",
    "inference.core.workflows.core_steps.transformations.byte_tracker.v2",
    "inference.core.workflows.core_steps.transformations.byte_tracker.v3",
    "inference.core.workflows.core_steps.transformations.camera_calibration.v1",
    "inference.core.workflows.core_steps.transformations.detection_offset.v1",
    "inference.core.workflows.core_steps.transformations.detections_combine.v1",
    "inference.core.workflows.core_steps.transformations.detections_filter.v1",
    "inference.core.workflows.core_steps.transformations.detections_merge.v1",
    "inference.core.workflows.core_steps.transformations.detections_transformation.v1",
    "inference.core.workflows.core_steps.transformations.dynamic_crop.v1",
    "inference.core.workflows.core_steps.transformations.dynamic_zones.v1",
    "inference.core.workflows.core_steps.transformations.geotag_detection.v1",
    "inference.core.workflows.core_steps.transformations.image_slicer.v1",
    "inference.core.workflows.core_steps.transformations.image_slicer.v2",
    "inference.core.workflows.core_steps.transformations.per_class_confidence_filter.v1",
    "inference.core.workflows.core_steps.transformations.perspective_correction.v1",
    "inference.core.workflows.core_steps.transformations.relative_static_crop.v1",
    "inference.core.workflows.core_steps.transformations.stabilize_detections.v1",
    "inference.core.workflows.core_steps.transformations.stitch_images.v1",
    "inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1",
    "inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v2",
    "inference.core.workflows.core_steps.transformations.track_class_lock.v1",
    "inference.core.workflows.core_steps.flow_control.continue_if.v1",
    "inference.core.workflows.core_steps.flow_control.delta_filter.v1",
    "inference.core.workflows.core_steps.flow_control.inner_workflow.v1",
    "inference.core.workflows.core_steps.flow_control.rate_limiter.v1",
    "inference.core.workflows.core_steps.flow_control.switch_case.v1",
    "inference.core.workflows.core_steps.visualizations.background_color.v1",
    "inference.core.workflows.core_steps.visualizations.blur.v1",
    "inference.core.workflows.core_steps.visualizations.bounding_box.v1",
    "inference.core.workflows.core_steps.visualizations.circle.v1",
    "inference.core.workflows.core_steps.visualizations.classification_label.v1",
    "inference.core.workflows.core_steps.visualizations.color.v1",
    "inference.core.workflows.core_steps.visualizations.corner.v1",
    "inference.core.workflows.core_steps.visualizations.crop.v1",
    "inference.core.workflows.core_steps.visualizations.dot.v1",
    "inference.core.workflows.core_steps.visualizations.ellipse.v1",
    "inference.core.workflows.core_steps.visualizations.grid.v1",
    "inference.core.workflows.core_steps.visualizations.halo.v1",
    "inference.core.workflows.core_steps.visualizations.halo.v2",
    "inference.core.workflows.core_steps.visualizations.heatmap.v1",
    "inference.core.workflows.core_steps.visualizations.icon.v1",
    "inference.core.workflows.core_steps.visualizations.keypoint.v1",
    "inference.core.workflows.core_steps.visualizations.label.v1",
    "inference.core.workflows.core_steps.visualizations.label.v2",
    "inference.core.workflows.core_steps.visualizations.line_zone.v1",
    "inference.core.workflows.core_steps.visualizations.model_comparison.v1",
    "inference.core.workflows.core_steps.visualizations.pixelate.v1",
    "inference.core.workflows.core_steps.visualizations.polygon.v1",
    "inference.core.workflows.core_steps.visualizations.polygon.v2",
    "inference.core.workflows.core_steps.visualizations.polygon_zone.v1",
    "inference.core.workflows.core_steps.visualizations.reference_path.v1",
    "inference.core.workflows.core_steps.visualizations.rich_label.v1",
    "inference.core.workflows.core_steps.visualizations.text_display.v1",
    "inference.core.workflows.core_steps.visualizations.trace.v1",
    "inference.core.workflows.core_steps.visualizations.triangle.v1",
    "inference.core.workflows.core_steps.fusion.buffer.v1",
    "inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1",
    "inference.core.workflows.core_steps.fusion.detections_consensus.v1",
    "inference.core.workflows.core_steps.fusion.detections_difference.v1",
    "inference.core.workflows.core_steps.fusion.detections_list_rollup.v1",
    "inference.core.workflows.core_steps.fusion.detections_stitch.v1",
    "inference.core.workflows.core_steps.fusion.dimension_collapse.v1",
    "inference.core.workflows.core_steps.fusion.frame_delay.v1",
    "inference.core.workflows.core_steps.fusion.image_stack.v1",
    "inference.core.workflows.core_steps.fusion.overlap_analysis.v1",
    "inference.core.workflows.core_steps.analytics.data_aggregator.v1",
    "inference.core.workflows.core_steps.analytics.detection_event_log.v1",
    "inference.core.workflows.core_steps.analytics.line_counter.v1",
    "inference.core.workflows.core_steps.analytics.line_counter.v2",
    "inference.core.workflows.core_steps.analytics.overlap.v1",
    "inference.core.workflows.core_steps.analytics.path_deviation.v1",
    "inference.core.workflows.core_steps.analytics.path_deviation.v2",
    "inference.core.workflows.core_steps.analytics.time_in_zone.v1",
    "inference.core.workflows.core_steps.analytics.time_in_zone.v2",
    "inference.core.workflows.core_steps.analytics.time_in_zone.v3",
    "inference.core.workflows.core_steps.analytics.velocity.v1",
    "inference.core.workflows.core_steps.math.cosine_similarity.v1",
    "inference.core.workflows.core_steps.sampling.identify_changes.v1",
    "inference.core.workflows.core_steps.sampling.identify_outliers.v1",
    "inference.core.workflows.core_steps.cache.cache_get.v1",
    "inference.core.workflows.core_steps.cache.cache_set.v1",
    "inference.core.workflows.core_steps.models.rknn.v1",
)

# Optional functionality is not advertised until its runtime dependency exists.
UNAVAILABLE_DEPENDENCIES = {
    "classical_cv/contrast_equalization": "scikit-image",
    "formatters/csv": "pandas",
    "transformations/qr_code_generator": "qrcode",
    "visualizations/mask": "pycocotools",
    "trackers": "trackers (use the original Supervision ByteTracker blocks)",
}


def load_numpy_blocks():
    from inference.core.workflows.prototypes.block import WorkflowBlock

    blocks = []
    for name in BLOCK_MODULES:
        module = import_module(name)
        blocks.extend(
            candidate
            for candidate in vars(module).values()
            if isclass(candidate)
            and candidate is not WorkflowBlock
            and issubclass(candidate, WorkflowBlock)
            and candidate.__module__ == name
        )
    return list(dict.fromkeys(blocks))
