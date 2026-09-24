#!/usr/bin/env python3
"""Normalize SDK wheel metadata and audit the ARM64 dependency closure."""

import argparse
import json
import re
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

from build_wheel import digest, write_wheel
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

FORBIDDEN_DISTRIBUTIONS = {
    "networkx",  # Use the explicit firmware-adapted distribution instead.
    "supervision",  # The upstream wheel would pull a second OpenCV distribution.
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "torch",
    "torchvision",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnx",
    "rknn-toolkit2",
    "inference-models",
    "transformers",
    "diffusers",
    "tensorflow",
}
FORBIDDEN_ROOTS = {
    "numpy",
    "numpy.libs",
    "cv2",
    "torch",
    "torchvision",
    "onnxruntime",
    "inference_models",
    "transformers",
    "diffusers",
    "tensorflow",
}


def normalize(path):
    with zipfile.ZipFile(path) as archive:
        wheel_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        )
        dist_info = wheel_name.rsplit("/", 1)[0]
        metadata = BytesParser().parsebytes(archive.read(wheel_name))
        original_tags = metadata.get_all("Tag", [])
        original_name = path.name
        original_hash = digest(path.read_bytes())
        prefix, python, abi, platform = path.stem.rsplit("-", 3)
        original_python = python
        stable_python = re.fullmatch(r"cp3(\d+)", python)
        if abi == "abi3" and stable_python and 2 <= int(stable_python[1]) <= 11:
            # A cp310-abi3 binary also supports CPython 3.11. Narrow its
            # declared interpreter set for this SDK's cp311-only installer;
            # never change a version-specific ABI or claim an older runtime.
            python = "cp311"
        if python == "py2.py3":
            if (
                abi != "none"
                or platform != "any"
                or metadata["Root-Is-Purelib"].lower() != "true"
            ):
                raise ValueError(f"Cannot normalize non-pure Python wheel: {path.name}")
            python = "py3"
        if platform == "any":
            if python not in {"py3", "cp311"} or abi != "none":
                raise ValueError(
                    f"SDK does not support this pure wheel tag: {path.name}"
                )
        elif (
            python not in {"py3", "cp311"}
            or (python == "py3" and abi != "none")
            or abi not in {"none", "cp311", "abi3"}
            or not all(
                component.endswith("_aarch64") for component in platform.split(".")
            )
        ):
            raise ValueError(f"SDK does not support this ARM64 wheel tag: {path.name}")
        tag = f"{python}-{abi}-{platform}"
        renamed = path.with_name(f"{prefix}-{tag}.whl")
        nonruntime_files = {
            name
            for name in archive.namelist()
            if ".data/" in name and not name.endswith("/")
        }
        if nonruntime_files - {"fonttools-4.65.0.data/data/share/man/man1/ttx.1"}:
            raise ValueError(
                f"Unsupported wheel .data layout: {sorted(nonruntime_files)}"
            )
        if original_tags == [tag] and renamed == path and not nonruntime_files:
            return path
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
            and not name.endswith(("/RECORD", "/RECORD.jws", "/RECORD.p7s"))
            and name not in nonruntime_files
        }
    del metadata["Tag"]
    metadata["Tag"] = tag
    # Keep the compound tag on one line. Some package validators read WHEEL
    # tags line-by-line rather than unfolding RFC email continuation lines.
    files[wheel_name] = metadata.as_bytes(policy=policy.compat32.clone(max_line_length=0))
    adaptation_name = dist_info + "/RV1126B_WHEEL_ADAPTATION.json"
    adaptation = (
        json.loads(files[adaptation_name])
        if adaptation_name in files
        else {
            "source_filename": original_name,
            "source_sha256": original_hash,
            "source_wheel_tags": original_tags,
            "device_wheel_tag": tag,
            "binary_abi_changed": False,
            "python_compatibility_narrowed": original_python != python and abi == "abi3",
            "reason": "SDK compound tag metadata equality and pure Python 3 selection",
        }
    )
    if nonruntime_files:
        adaptation["removed_nonruntime_files"] = sorted(nonruntime_files)
        adaptation["nonruntime_removal_reason"] = (
            "SDK disallows .data; remove only the fonttools ttx manual page"
        )
    files[adaptation_name] = json.dumps(
        adaptation,
        sort_keys=True,
        indent=2,
    ).encode()
    write_wheel(renamed, files, dist_info)
    if renamed != path:
        path.unlink()
    return renamed


def audit(directory):
    distributions = {}
    wheels = []
    for path in sorted(directory.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            metadata_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            distribution = canonicalize_name(metadata["Name"])
            if distribution in FORBIDDEN_DISTRIBUTIONS or distribution.startswith(
                "nvidia-"
            ):
                raise ValueError(f"Forbidden bundled distribution: {distribution}")
            if distribution in distributions:
                raise ValueError(
                    f"Duplicate distribution in wheelhouse: {distribution}"
                )
            for name in archive.namelist():
                if ".data/" in name:
                    raise ValueError(f"SDK does not support wheel .data layout: {name}")
                if name.split("/")[0] in FORBIDDEN_ROOTS:
                    raise ValueError(
                        f"Wheel would override a system/forbidden package: {name}"
                    )
                if re.search(r"\.so(?:\.|$)", name):
                    data = archive.read(name)
                    if data[:4] == b"\x7fELF" and (
                        data[4] != 2
                        or data[5] != 1
                        or int.from_bytes(data[18:20], "little") != 183
                    ):
                        raise ValueError(f"Non-AArch64 ELF in {path.name}: {name}")
            distributions[distribution] = {
                "version": metadata["Version"],
                "requires_dist": metadata.get_all("Requires-Dist", []),
            }
            wheels.append(
                {
                    "file": path.name,
                    "sha256": digest(path.read_bytes()),
                    "bytes": path.stat().st_size,
                    "expanded_bytes": sum(
                        info.file_size for info in archive.infolist()
                    ),
                }
            )
    available = {name: entry["version"] for name, entry in distributions.items()}
    available["numpy"] = "1.23.5"
    environment = {
        **default_environment(),
        "python_version": "3.11",
        "python_full_version": "3.11.6",
        "platform_machine": "aarch64",
        "platform_system": "Linux",
        "sys_platform": "linux",
        "extra": "",
    }
    for name, entry in distributions.items():
        for raw in entry["requires_dist"]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in available or not requirement.specifier.contains(
                Version(available[dependency])
            ):
                raise ValueError(f"Unresolved device dependency: {name} requires {raw}")
    result = {
        "target": {
            "python": "3.11",
            "architecture": "aarch64",
            "firmware_numpy": "1.23.5",
            "firmware_opencv": "4.6.0",
        },
        "compressed_bytes": sum(wheel["bytes"] for wheel in wheels),
        "expanded_bytes": sum(wheel["expanded_bytes"] for wheel in wheels),
        "wheels": wheels,
        "distributions": distributions,
    }
    (directory / "wheelhouse-audit.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    (directory / "SHA256SUMS").write_text(
        "".join(f"{wheel['sha256']}  {wheel['file']}\n" for wheel in wheels)
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("target", "compressed_bytes", "expanded_bytes")
            },
            indent=2,
        )
    )
    print(f"Audited {len(wheels)} wheels; no unresolved runtime dependencies")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--normalize", action="store_true")
    arguments = parser.parse_args()
    if arguments.normalize:
        for path in sorted(arguments.directory.glob("*.whl")):
            normalize(path)
    audit(arguments.directory)
