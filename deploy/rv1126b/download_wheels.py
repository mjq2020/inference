#!/usr/bin/env python3
"""Download pinned ARM64 wheels without downloading firmware NumPy/OpenCV."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from build_wheel import (
    DEFAULT_FONTS_DIR,
    ROOT,
    build_networkx_wheel,
    build_source_wheel,
    build_supervision_wheel,
    requirements,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument("--fonts-dir", type=Path, default=DEFAULT_FONTS_DIR)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # Do not inherit arbitrary host package mirrors, CUDA indexes or pip config.
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("PIP_")
    }
    env["PIP_CONFIG_FILE"] = os.devnull
    command = [
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "download",
        "--index-url",
        arguments.index_url,
        "--only-binary=:all:",
        "--no-deps",
        "--python-version=311",
        "--implementation=cp",
        "--abi=cp311",
        "--abi=abi3",
        "--abi=none",
        "--platform=manylinux_2_28_aarch64",
        "--platform=manylinux_2_27_aarch64",
        "--platform=manylinux2014_aarch64",
        "--platform=manylinux_2_17_aarch64",
    ]
    deps = [
        item
        for item in requirements()
        if not item.startswith(("supervision-rv1126b==", "networkx-rv1126b=="))
    ]
    subprocess.run(
        command + ["--dest", str(arguments.output), *deps], env=env, check=True
    )
    with tempfile.TemporaryDirectory(prefix="supervision-source-") as temporary:
        subprocess.run(
            command + ["--dest", temporary, "supervision==0.29.1"], env=env, check=True
        )
        source = next(Path(temporary).glob("supervision-*.whl"))
        build_supervision_wheel(source, arguments.output)
        subprocess.run(
            command + ["--dest", temporary, "networkx==3.4.2"], env=env, check=True
        )
        build_networkx_wheel(
            next(Path(temporary).glob("networkx-*.whl")), arguments.output
        )
    # Use the repository's immutable URL + SHA registry for all 20 fonts and
    # their licenses. Network access is confined to dependency preparation;
    # build_source_wheel itself only consumes verified local assets.
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "build_scripts/download_fonts.py"),
            "--target-dir",
            str(arguments.fonts_dir),
        ],
        env=env,
        check=True,
    )
    build_source_wheel(arguments.output, fonts_dir=arguments.fonts_dir)


if __name__ == "__main__":
    main()
