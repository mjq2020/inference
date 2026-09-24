#!/usr/bin/env python3
"""Compatibility launcher; the managed application is maintained in ext."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    builder = args.sdk_root.resolve() / "apps/inference-rv1126b/build.py"
    if not builder.is_file():
        parser.error("Update recamera-pro-ext-api: apps/inference-rv1126b/build.py is required")
    subprocess.run(
        [sys.executable, str(builder), "--sdk-root", str(args.sdk_root.resolve()),
         "--engine-source", str(Path(__file__).resolve().parents[2]), *remaining],
        check=True,
    )


if __name__ == "__main__":
    main()
