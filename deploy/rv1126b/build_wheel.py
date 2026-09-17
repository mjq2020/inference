#!/usr/bin/env python3
"""Build deterministic device wheels without executing the full setup.py."""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import runpy
import zipfile
from email.parser import BytesParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "0.3.8"
SUPERVISION_VERSION = "0.29.1"
NETWORKX_VERSION = "3.4.2"
FONTS_PACKAGE = Path("inference/core/workflows/core_steps/visualizations/common/fonts")
DEFAULT_FONTS_DIR = ROOT / FONTS_PACKAGE / "assets"
FORBIDDEN_PATHS = (
    "inference_models/",
    "inference/models/",
    "inference/core/models/",
    "inference/enterprise/",
    "inference/core/interfaces/",
    "inference_sdk/http/client.py",
    "tests/",
)


def requirements():
    return [
        line.strip()
        for line in (HERE / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_wheel(destination, files, dist_info):
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([name, "sha256=" + checksum.decode(), len(data)])
    writer.writerow([dist_info + "/RECORD", "", ""])
    files[dist_info + "/RECORD"] = record.getvalue().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return destination


def font_registry():
    # The upstream registry is intentionally stdlib-only. Do not import inference
    # during packaging: its default runtime may load optional host model packages.
    return runpy.run_path(str(ROOT / FONTS_PACKAGE / "registry.py"))["FONTS_REGISTRY"]


def collect_font_assets(fonts_dir):
    """Require all pinned font binaries and license texts; never download at build."""
    files = {}
    provenance = {}
    fonts_dir = fonts_dir.resolve()
    for identifier, metadata in sorted(font_registry().items()):
        for filename, expected, url in (
            (metadata.file_name, metadata.sha256, metadata.source_url),
            ("OFL.txt", metadata.license_sha256, metadata.license_url),
        ):
            relative = Path(identifier) / filename
            path = fonts_dir / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(fonts_dir)
            ):
                raise ValueError(
                    f"Missing regular approved font asset: {path}. "
                    "Run build_scripts/download_fonts.py --target-dir <fonts-dir> "
                    "before building the offline wheel."
                )
            data = path.read_bytes()
            if digest(data) != expected:
                raise ValueError(f"Approved font asset checksum mismatch: {path}")
            name = (FONTS_PACKAGE / "assets" / relative).as_posix()
            files[name] = data
            provenance[name] = {"sha256": expected, "source_url": url}
    return files, provenance


def build_source_wheel(output, fonts_dir=None):
    files = {}
    for line in (HERE / "source_files.txt").read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Source must be a regular repository file: {name}")
        if name.startswith(FORBIDDEN_PATHS) or "_tensor" in Path(name).stem:
            raise ValueError(
                f"Excluded model/host implementation in source list: {name}"
            )
        files[name] = path.read_bytes()
    catalog = runpy.run_path(
        str(ROOT / "inference/core/workflows/core_steps/catalog_rv1126b.py")
    )["BLOCK_MODULES"]
    for module in catalog:
        source = module.replace(".", "/") + ".py"
        if source not in files:
            raise ValueError(
                f"Catalog implementation absent from source_files.txt: {source}"
            )
    # Edge modules evolve together; legacy packages are never auto-discovered.
    for path in sorted((ROOT / "inference/edge").glob("*.py")):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    # Only this reviewed lightweight frontend belongs in the device package.
    # Serving it needs no Node process or legacy HTTP/model dependencies.
    for name in (
        "index.html",
        "build.html",
        "app.css",
        "common.js",
        "app.js",
        "builder.js",
        "device.html",
        "device.js",
    ):
        path = ROOT / "inference/edge/static" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Required browser asset is missing or not a regular file: {name}"
            )
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    font_files, font_provenance = collect_font_assets(fonts_dir or DEFAULT_FONTS_DIR)
    files.update(font_files)
    font_readme = FONTS_PACKAGE / "README.md"
    files[font_readme.as_posix()] = (ROOT / font_readme).read_bytes()
    dist_info = f"inference_rv1126b-{VERSION}.dist-info"
    metadata = [
        "Metadata-Version: 2.4",
        "Name: inference-rv1126b",
        f"Version: {VERSION}",
        "Summary: RV1126B RKNN inference and NumPy Workflow application",
        "Requires-Python: >=3.11,<3.12",
        "License-Expression: Apache-2.0",
        "License-File: LICENSE",
        "License-File: LICENSE.core",
        "Requires-External: NumPy (=1.23.5, supplied by firmware)",
        "Requires-External: OpenCV (=4.6.0, supplied by firmware)",
        "Requires-External: reCamera kit RemoteRknnSession and FrameSource",
    ]
    metadata += [f"Requires-Dist: {requirement}" for requirement in requirements()]
    files[dist_info + "/METADATA"] = ("\n".join(metadata) + "\n\n").encode()
    files[dist_info + "/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: rv1126b-build\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode()
    files[dist_info + "/entry_points.txt"] = (
        "[console_scripts]\ninference-rv1126b = inference_edge:main\n"
    ).encode()
    for name in ["LICENSE", "LICENSE.core"]:
        files[dist_info + "/licenses/" + name] = (ROOT / name).read_bytes()
    files[dist_info + "/FONT_ASSETS.json"] = json.dumps(
        font_provenance, sort_keys=True, indent=2
    ).encode()
    # Include every installed payload, including font licenses and provenance.
    # RECORD and this manifest itself are generated afterward.
    files[dist_info + "/SOURCE_MANIFEST.json"] = json.dumps(
        {name: digest(data) for name, data in sorted(files.items())},
        sort_keys=True,
        indent=2,
    ).encode()
    return write_wheel(
        output / f"inference_rv1126b-{VERSION}-py3-none-any.whl", files, dist_info
    )


def build_supervision_wheel(source, output):
    """Give the platform adaptation its own honest distribution identity."""
    with zipfile.ZipFile(source) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        if (
            metadata["Name"] != "supervision"
            or metadata["Version"] != SUPERVISION_VERSION
        ):
            raise ValueError("Expected the original supervision==0.29.1 wheel")
        old_info = metadata_path.rsplit("/", 1)[0]
        new_info = f"supervision_rv1126b-{SUPERVISION_VERSION}.dist-info"
        files = {}
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(
                ("/RECORD", "/RECORD.jws", "/RECORD.p7s")
            ):
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Invalid path in supervision wheel")
            files[name.replace(old_info + "/", new_info + "/", 1)] = archive.read(name)
    metadata.replace_header("Name", "supervision-rv1126b")
    dependencies = metadata.get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    removed = []
    for dependency in dependencies:
        if re.match(r"opencv[-_]python(?:[<>=!~ ;\[]|$)", dependency, re.I):
            removed.append(dependency)
        else:
            metadata["Requires-Dist"] = dependency
    if removed != ["opencv-python>=4.5.5.64"]:
        raise ValueError(f"Unexpected upstream OpenCV requirement: {removed}")
    metadata["Requires-External"] = "OpenCV (=4.6.0, supplied by RV1126B firmware)"
    files[new_info + "/METADATA"] = metadata.as_bytes()
    files[new_info + "/RV1126B_ADAPTATION.json"] = json.dumps(
        {
            "source_wheel": source.name,
            "source_sha256": digest(source.read_bytes()),
            "source_distribution": "supervision==0.29.1",
            "removed_requires_dist": removed,
            "replacement": "Firmware OpenCV 4.6.0; checked by platform_probe.py",
            "python_sources_changed": False,
        },
        sort_keys=True,
        indent=2,
    ).encode()
    return write_wheel(
        output / f"supervision_rv1126b-{SUPERVISION_VERSION}-py3-none-any.whl",
        files,
        new_info,
    )


def build_networkx_wheel(source, output):
    """Make compressed graph IO optional on Python builds without _bz2."""
    with zipfile.ZipFile(source) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        if metadata["Name"] != "networkx" or metadata["Version"] != NETWORKX_VERSION:
            raise ValueError("Expected the original networkx==3.4.2 wheel")
        old_info = metadata_path.rsplit("/", 1)[0]
        new_info = f"networkx_rv1126b-{NETWORKX_VERSION}.dist-info"
        files = {
            name.replace(old_info + "/", new_info + "/", 1): archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
            and not name.endswith(("/RECORD", "/RECORD.jws", "/RECORD.p7s"))
        }
    changed_file = "networkx/utils/decorators.py"
    original_source = files[changed_file]
    text = original_source.decode()
    if not text.startswith("import bz2\n") or text.count('".bz2": bz2.BZ2File,') != 1:
        raise ValueError(
            "NetworkX compressed graph IO source differs from audited version"
        )
    text = text.removeprefix("import bz2\n")
    helper = """def _rv1126b_open_bz2(path, mode):
    try:
        from bz2 import BZ2File
    except ImportError as error:
        raise RuntimeError(
            "Bzip2 graph IO is unavailable: this firmware Python has no _bz2. "
            "Use an uncompressed graph file."
        ) from error
    return BZ2File(path, mode)


"""
    text = text.replace("fopeners = {", helper + "fopeners = {", 1)
    text = text.replace('".bz2": bz2.BZ2File,', '".bz2": _rv1126b_open_bz2,')
    files[changed_file] = text.encode()
    metadata.replace_header("Name", "networkx-rv1126b")
    files[new_info + "/METADATA"] = metadata.as_bytes()
    files[new_info + "/RV1126B_ADAPTATION.json"] = json.dumps(
        {
            "source_wheel": source.name,
            "source_sha256": digest(source.read_bytes()),
            "source_distribution": "networkx==3.4.2",
            "changed_files": {
                changed_file: {
                    "source_sha256": digest(original_source),
                    "derived_sha256": digest(files[changed_file]),
                }
            },
            "reason": "Import bz2 only for .bz2 graph IO; fail explicitly when firmware lacks _bz2",
        },
        sort_keys=True,
        indent=2,
    ).encode()
    return write_wheel(
        output / f"networkx_rv1126b-{NETWORKX_VERSION}-py3-none-any.whl",
        files,
        new_info,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision-wheel", type=Path)
    parser.add_argument("--networkx-wheel", type=Path)
    parser.add_argument("--fonts-dir", type=Path, default=DEFAULT_FONTS_DIR)
    arguments = parser.parse_args()
    wheels = [build_source_wheel(arguments.output, fonts_dir=arguments.fonts_dir)]
    if arguments.supervision_wheel:
        wheels.append(
            build_supervision_wheel(arguments.supervision_wheel, arguments.output)
        )
    if arguments.networkx_wheel:
        wheels.append(build_networkx_wheel(arguments.networkx_wheel, arguments.output))
    for path in wheels:
        print(f"{digest(path.read_bytes())}  {path}")


if __name__ == "__main__":
    main()
