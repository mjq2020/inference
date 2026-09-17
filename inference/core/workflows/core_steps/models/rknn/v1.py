"""RKNN execution for the original object-detection Workflow manifests.

Only model execution changes. Parameter defaults, selectors and output kinds
come from the upstream v1/v2/v3 manifests, so imported workflows keep their
meaning when the device catalog grows.
"""

from typing import Any, List, Literal, Optional, Type

import cv2

from inference.core.env import DEFAULT_CONFIDENCE
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    BlockManifest as OriginalManifestV1,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v2 import (
    BlockManifest as OriginalManifestV2,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as OriginalManifestV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class LocalModelResources:
    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        # The model ID names an installed RKNN package, not a cloud download.
        # Model metadata is not a platform NPU permission grant.
        return []


class BlockManifest(LocalModelResources, OriginalManifestV1):
    type: Literal[
        "roboflow_core/roboflow_object_detection_model@v1",
        "RoboflowObjectDetectionModel",
        "ObjectDetectionModel",
        "rv1126b/rknn_object_detection@v1",
    ]


class BlockManifestV2(LocalModelResources, OriginalManifestV2):
    pass


class BlockManifestV3(LocalModelResources, OriginalManifestV3):
    pass


class RknnObjectDetectionBlockV1(WorkflowBlock):
    def __init__(self, model_manager: Any):
        self._model_manager = model_manager

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        confidence: float,
        class_filter: Optional[List[str]],
        iou_threshold: float,
        max_detections: int,
        class_agnostic_nms: Optional[bool],
        max_candidates: int,
        disable_active_learning: bool,
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        if not disable_active_learning:
            raise ValueError(
                "Active learning is unavailable in the RV1126B inference runtime; "
                "set disable_active_learning=true"
            )
        # The target is dead configuration when active learning is disabled,
        # as in the original block. Do not reject an otherwise valid graph.
        from inference.edge.limits import reserve_model_call

        results = []
        for image in images:
            reserve_model_call()
            pixels = image.numpy_image
            if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[2] == 1):
                # Match the original image loader at the model boundary;
                # grayscale Workflow outputs themselves stay two-dimensional.
                pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_id,
                request={
                    "model_id": model_id,
                    "image": {"type": "numpy", "value": pixels},
                    "confidence": confidence,
                    # Original response filtering follows NMS. Filtering first
                    # changes class-agnostic suppression and max_detections.
                    "class_filter": None,
                    "iou_threshold": iou_threshold,
                    "max_detections": max_detections,
                    "class_agnostic_nms": class_agnostic_nms,
                    "max_candidates": max_candidates,
                },
            )
            results.append(dict(prediction))
        detections = convert_inference_detections_batch_to_sv_detections(results)
        detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=detections, prediction_type="object-detection"
        )
        detections = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=detections, classes_to_accept=class_filter
        )
        detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images, predictions=detections
        )
        return [
            {"inference_id": raw.get("inference_id"), "predictions": prediction}
            for raw, prediction in zip(results, detections)
        ]


class RknnObjectDetectionBlockV2(RknnObjectDetectionBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifestV2

    def run(self, *, model_id: str, **kwargs) -> BlockResult:
        return [
            {**result, "model_id": model_id}
            for result in super().run(model_id=model_id, **kwargs)
        ]


class RknnObjectDetectionBlockV3(RknnObjectDetectionBlockV2):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifestV3

    def run(
        self,
        *,
        confidence_mode: str,
        custom_confidence: Optional[float],
        **kwargs,
    ) -> BlockResult:
        if confidence_mode == "custom":
            if custom_confidence is None:
                raise ValueError("custom_confidence is required for custom mode")
            confidence = custom_confidence
        elif confidence_mode in {"best", "default"}:
            # Current RKNN packages have no RecommendedParameters. The original
            # ConfidenceFilter falls back to the model default when evaluation
            # recommendations are absent, including in best mode. Both supported
            # YOLO decoder families use the original 0.4 default.
            confidence = DEFAULT_CONFIDENCE
        else:
            raise ValueError("confidence_mode must be best, default, or custom")
        return super().run(confidence=confidence, **kwargs)
