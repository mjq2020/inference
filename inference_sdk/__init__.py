import os
import warnings

from inference_sdk.config import (
    InferenceSDKDeprecationWarning,
    InferenceSDKGuidanceWarning,
)

# The device workflow engine imports shared request types from this package.
# Those imports must not also instantiate the optional remote HTTP/video stack.
# Keep the standalone SDK independent of the inference server distribution.
if os.getenv("INFERENCE_RUNTIME_PROFILE", "full").strip().lower() != "rv1126b":
    from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.http.entities import (
    ApiKeyTransport,
    InferenceConfiguration,
    VisualisationResponseFormat,
)
from inference_sdk.utils.environment import str2bool

# Environment variable to control whether SDK warnings are disabled.
# Set to "true" to disable all SDK-specific warnings, "false" to enable them.
# Default is "false" (warnings enabled).
INFERENCE_WARNINGS_DISABLED = str2bool(
    os.getenv("INFERENCE_WARNINGS_DISABLED", "False")
)

if INFERENCE_WARNINGS_DISABLED:
    warnings.simplefilter("ignore", InferenceSDKDeprecationWarning)
    warnings.simplefilter("ignore", InferenceSDKGuidanceWarning)

try:
    from inference_sdk.version import __version__
except ImportError:
    __version__ = "development"


def __getattr__(name):
    if name == "InferenceHTTPClient":
        from inference_sdk.http.client import InferenceHTTPClient

        globals()[name] = InferenceHTTPClient
        return InferenceHTTPClient
    raise AttributeError(f"module 'inference_sdk' has no attribute {name!r}")
