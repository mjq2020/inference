"""Select the process runtime before importing model or workflow dependencies."""

import os

from dotenv import dotenv_values


def _runtime_profile() -> str:
    profile = os.environ.get("INFERENCE_RUNTIME_PROFILE")
    if profile is None:
        profile = dotenv_values(os.path.join(os.getcwd(), ".env")).get(
            "INFERENCE_RUNTIME_PROFILE", "full"
        )
    if not isinstance(profile, str) or profile.strip().lower() not in {
        "full",
        "rv1126b",
    }:
        raise ValueError("INFERENCE_RUNTIME_PROFILE must be 'full' or 'rv1126b'.")
    profile = profile.strip().lower()
    # Child processes must inherit the selected profile even when their working
    # directory has no .env. Profile changes require restarting the interpreter.
    os.environ["INFERENCE_RUNTIME_PROFILE"] = profile
    return profile


INFERENCE_RUNTIME_PROFILE = _runtime_profile()
IS_RV1126B = INFERENCE_RUNTIME_PROFILE == "rv1126b"
