#!/usr/bin/env python3
"""Read-only check of the firmware Python/image/NPU-client contract."""

import argparse
import json
import platform
import sys


def probe(require_device=False):
    import cv2
    import numpy

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This distribution requires CPython 3.11")
    if numpy.__version__ != "1.23.5" or cv2.__version__ != "4.6.0":
        raise RuntimeError("Expected firmware NumPy 1.23.5 and OpenCV 4.6.0")
    result = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "numpy_path": numpy.__file__,
        "opencv": cv2.__version__,
        "opencv_path": cv2.__file__,
    }
    if require_device:
        if platform.machine() != "aarch64":
            raise RuntimeError("Device installation requires aarch64")
        from kit.adapters.official import OfficialFrameSource
        from kit.runtime.remote import RemoteRknnSession

        result["kit_frame_source"] = OfficialFrameSource.__module__
        result["kit_rknn_session"] = RemoteRknnSession.__module__
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-device", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(probe(arguments.require_device), indent=2))
