"""Import only the supported NumPy blocks in the RV1126B distribution.

This catalogue uses the existing compiler, scheduler, block manifests and
serializers. Unsupported blocks fail workflow compilation before model loading.
"""

from typing import List, Type

from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_action_recognition_prediction_kind,
    deserialize_boolean_kind,
    deserialize_bytes_kind,
    deserialize_classification_prediction_kind,
    deserialize_detections_kind,
    deserialize_dictionary_kind,
    deserialize_float_kind,
    deserialize_float_zero_to_one_kind,
    deserialize_image_kind,
    deserialize_integer_kind,
    deserialize_labeled_points_kind,
    deserialize_list_of_values_kind,
    deserialize_numpy_array,
    deserialize_optional_string_kind,
    deserialize_point_kind,
    deserialize_rgb_color_kind,
    deserialize_rle_detections_kind,
    deserialize_string_kind,
    deserialize_timestamp,
    deserialize_video_metadata_kind,
    deserialize_zone_kind,
)
from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialise_rle_sv_detections,
    serialise_sv_detections,
    serialize_action_recognition_prediction_kind,
    serialize_secret,
    serialize_timestamp,
    serialize_video_metadata_kind,
    serialize_wildcard_kind,
)
from inference.core.workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
    BAR_CODE_DETECTION_KIND,
    BOOLEAN_KIND,
    BYTES_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    CONTOURS_KIND,
    DETECTION_KIND,
    DETECTIONS_OVERLAPS_KIND,
    DICTIONARY_KIND,
    EMBEDDING_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    INFERENCE_ID_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LABELED_POINTS_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    POINT_KIND,
    PREDICTION_TYPE_KIND,
    QR_CODE_DETECTION_KIND,
    RGB_COLOR_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MANAGED_KEY,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_SOLUTION_KIND,
    SECRET_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    SERIALISED_PAYLOADS_KIND,
    STRING_KIND,
    TIMESTAMP_KIND,
    TOP_CLASS_KIND,
    VIDEO_METADATA_KIND,
    WILDCARD_KIND,
    ZONE_KIND,
    Kind,
)
from inference.core.workflows.prototypes.block import StepExecutionMode, WorkflowBlock

KINDS_SERIALIZERS = {
    IMAGE_KIND.name: serialise_image,
    VIDEO_METADATA_KIND.name: serialize_video_metadata_kind,
    ACTION_RECOGNITION_PREDICTION_KIND.name: serialize_action_recognition_prediction_kind,
    OBJECT_DETECTION_PREDICTION_KIND.name: serialise_sv_detections,
    INSTANCE_SEGMENTATION_PREDICTION_KIND.name: serialise_sv_detections,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.name: serialise_rle_sv_detections,
    KEYPOINT_DETECTION_PREDICTION_KIND.name: serialise_sv_detections,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND.name: serialise_rle_sv_detections,
    QR_CODE_DETECTION_KIND.name: serialise_sv_detections,
    BAR_CODE_DETECTION_KIND.name: serialise_sv_detections,
    SECRET_KIND.name: serialize_secret,
    WILDCARD_KIND.name: serialize_wildcard_kind,
    DETECTIONS_OVERLAPS_KIND.name: serialize_wildcard_kind,
    TIMESTAMP_KIND.name: serialize_timestamp,
}

KINDS_DESERIALIZERS = {
    IMAGE_KIND.name: deserialize_image_kind,
    VIDEO_METADATA_KIND.name: deserialize_video_metadata_kind,
    ACTION_RECOGNITION_PREDICTION_KIND.name: deserialize_action_recognition_prediction_kind,
    OBJECT_DETECTION_PREDICTION_KIND.name: deserialize_detections_kind,
    INSTANCE_SEGMENTATION_PREDICTION_KIND.name: deserialize_detections_kind,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.name: deserialize_rle_detections_kind,
    KEYPOINT_DETECTION_PREDICTION_KIND.name: deserialize_detections_kind,
    QR_CODE_DETECTION_KIND.name: deserialize_detections_kind,
    BAR_CODE_DETECTION_KIND.name: deserialize_detections_kind,
    NUMPY_ARRAY_KIND.name: deserialize_numpy_array,
    ROBOFLOW_MODEL_ID_KIND.name: deserialize_string_kind,
    ROBOFLOW_PROJECT_KIND.name: deserialize_string_kind,
    ROBOFLOW_SOLUTION_KIND.name: deserialize_string_kind,
    ROBOFLOW_API_KEY_KIND.name: deserialize_optional_string_kind,
    ROBOFLOW_MANAGED_KEY.name: deserialize_optional_string_kind,
    FLOAT_ZERO_TO_ONE_KIND.name: deserialize_float_zero_to_one_kind,
    LIST_OF_VALUES_KIND.name: deserialize_list_of_values_kind,
    DETECTIONS_OVERLAPS_KIND.name: deserialize_list_of_values_kind,
    BOOLEAN_KIND.name: deserialize_boolean_kind,
    INTEGER_KIND.name: deserialize_integer_kind,
    STRING_KIND.name: deserialize_string_kind,
    TOP_CLASS_KIND.name: deserialize_string_kind,
    FLOAT_KIND.name: deserialize_float_kind,
    DICTIONARY_KIND.name: deserialize_dictionary_kind,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND.name: deserialize_rle_detections_kind,
    CLASSIFICATION_PREDICTION_KIND.name: deserialize_classification_prediction_kind,
    POINT_KIND.name: deserialize_point_kind,
    LABELED_POINTS_KIND.name: deserialize_labeled_points_kind,
    ZONE_KIND.name: deserialize_zone_kind,
    RGB_COLOR_KIND.name: deserialize_rgb_color_kind,
    LANGUAGE_MODEL_OUTPUT_KIND.name: deserialize_string_kind,
    PREDICTION_TYPE_KIND.name: deserialize_string_kind,
    PARENT_ID_KIND.name: deserialize_string_kind,
    BYTES_KIND.name: deserialize_bytes_kind,
    INFERENCE_ID_KIND.name: deserialize_string_kind,
    TIMESTAMP_KIND.name: deserialize_timestamp,
}

REGISTERED_INITIALIZERS = {
    "api_key": None,
    "step_execution_mode": StepExecutionMode.LOCAL,
    "background_tasks": None,
    "thread_pool_executor": None,
    "disable_sinks": True,
    "allow_access_to_file_system": False,
    "allow_access_to_environmental_variables": False,
}


def load_kinds() -> List[Kind]:
    # Kinds are metadata, not model implementations. Include the original set
    # needed by the expanded catalog without importing tensor-native carriers.
    from inference.core.workflows.execution_engine.entities import types

    return [value for value in vars(types).values() if isinstance(value, Kind)]


def load_blocks() -> List[Type[WorkflowBlock]]:
    from inference.core.workflows.core_steps.catalog_rv1126b import load_numpy_blocks

    return load_numpy_blocks()
