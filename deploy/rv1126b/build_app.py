#!/usr/bin/env python3
"""Build a manifest-v2 offline app; signatures remain a publisher operation."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wheel_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"Wheel must be a regular .whl file: {path}")
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(f"Invalid wheel metadata: {path.name}")
        if archive.getinfo(metadata_paths[0]).file_size > 1024 * 1024:
            raise ValueError(f"Wheel metadata exceeds 1 MiB: {path.name}")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if (
            len(metadata.get_all("Name", [])) != 1
            or len(metadata.get_all("Version", [])) != 1
        ):
            raise ValueError(f"Wheel must declare one Name and Version: {path.name}")
    parts = path.name[:-4].split("-")
    if len(parts) != 5:
        raise ValueError(f"Unsupported wheel filename: {path.name}")
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "filename": path.name,
        "file": f"wheels/{path.name}",
        "sha256": digest(path),
        "size": path.stat().st_size,
        "tags": ["-".join(parts[-3:])],
        "source": "bundled",
    }


def load_sdk(sdk_root):
    """Use the publisher and installer validators from the requested SDK."""
    market = sdk_root / "market"
    spec = importlib.util.spec_from_file_location(
        "recamera_package_build", market / "packaging" / "build.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("SDK root does not contain market/packaging/build.py")
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        sys.path.insert(0, str(market))
        spec.loader.exec_module(module)
        from appmgr import manifest, paths, pythonenv

        if not Path(manifest.__file__).resolve().is_relative_to(market):
            raise ValueError(
                "A different SDK appmgr is already imported; run the builder in a fresh process"
            )
    finally:
        sys.path[:] = before
    return SimpleNamespace(
        build=module.build, manifest=manifest, paths=paths, pythonenv=pythonenv
    )


def validate_wheels(sdk, wheels, descriptors):
    """Catch target install failures without executing AArch64 wheel code."""
    owners, expanded = {}, [0]
    for index, (path, descriptor) in enumerate(zip(wheels, descriptors)):
        sdk.manifest._validate_wheel(descriptor, index)
        project = sdk.pythonenv._normalise_project(descriptor["name"])
        if project in sdk.pythonenv._PLATFORM_PROJECTS:
            raise ValueError(f"Cannot replace platform-owned project: {project}")
        sdk.pythonenv._validate_wheel_archive(descriptor, str(path), owners, expanded)
    protected_modules = {"kit", "recamera_ext", "rknnlite", "jinja2", "markupsafe"}
    for member in owners:
        if member.split("/", 1)[0].split(".", 1)[0] in protected_modules:
            raise ValueError(f"Wheel shadows a platform-owned module: {member}")
    if "inference_edge_probe.py" not in owners:
        raise ValueError(
            "Source wheel does not provide the manifest import probe inference_edge_probe"
        )
    return expanded[0]


def build_app(sdk_root, wheelhouse, model_root, out):
    sdk_root, wheelhouse, model_root, out = (
        Path(value).resolve() for value in (sdk_root, wheelhouse, model_root, out)
    )
    sdk = load_sdk(sdk_root)

    # Import only after selecting rv1126b; validates the same schema used at runtime.
    os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from inference.edge.models import LocalModelStore

    # Reuse the ARM64 dependency/forbidden-package audit. This does not
    # normalize or mutate wheel archives; preparation owns such changes.
    before = list(sys.path)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from audit_wheels import audit

        with contextlib.redirect_stdout(io.StringIO()):
            wheel_audit = audit(wheelhouse)
    finally:
        sys.path[:] = before

    wheels = sorted(wheelhouse.glob("*.whl"))
    if not any(path.name.startswith("inference_rv1126b-") for path in wheels):
        raise ValueError("wheelhouse must contain the inference_rv1126b source wheel")
    store = LocalModelStore(model_root)
    metadata_files = sorted(model_root.rglob("model.json"))
    if not metadata_files:
        raise ValueError("Provide at least one validated RKNN model package")
    descriptors = [
        store.get(path.parent.relative_to(model_root).as_posix())
        for path in metadata_files
    ]
    template = json.loads(
        (Path(__file__).parent / "manifest.template.json").read_text()
    )
    template["python"]["wheels"] = [wheel_record(path) for path in wheels]
    environment_bytes = validate_wheels(sdk, wheels, template["python"]["wheels"])
    template["models"] = []
    template["artifacts"] = []
    with tempfile.TemporaryDirectory(prefix="inference-app-") as temporary:
        app = Path(temporary) / "app"
        app.mkdir()
        shutil.copyfile(Path(__file__).parent / "app.py", app / "app.py")
        shutil.copyfile(Path(__file__).parent / "icon.jpg", app / "icon.jpg")
        (app / "wheels").mkdir()
        for path in wheels:
            shutil.copyfile(path, app / "wheels" / path.name)
        for index, descriptor in enumerate(descriptors):
            relative = Path("models") / descriptor.path.relative_to(
                model_root.resolve()
            )
            target = app / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(descriptor.path, target)
            source_metadata = model_root / descriptor.model_id / "model.json"
            metadata_target = app / "models" / descriptor.model_id / "model.json"
            metadata_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_metadata, metadata_target)
            template["artifacts"].append(
                {
                    "id": f"detector-{index}",
                    "kind": "rknn",
                    "source": "bundled",
                    "file": relative.as_posix(),
                    "mount": relative.as_posix(),
                    "sha256": descriptor.sha256,
                    "size": descriptor.path.stat().st_size,
                    "required": True,
                    "share_scope": "content",
                }
            )
            template["models"].append(
                {
                    "id": f"detector-{index}",
                    "file": relative.as_posix(),
                    "task": "detect",
                    "input": list(descriptor.input.shape),
                    "classes": list(descriptor.labels),
                }
            )
        (app / "manifest.json").write_text(json.dumps(template, indent=2) + "\n")
        # Revalidate the staged metadata/binary pair after copying, before
        # producing a BOM for an asset that could have changed during staging.
        staged_store = LocalModelStore(app / "models")
        for descriptor in descriptors:
            staged_store.get(descriptor.model_id)
        sdk.manifest.validate_manifest(template, allow_v1=False)
        source_files = [p for p in app.rglob("*") if p.is_file()]
        payload_bytes = sum(p.stat().st_size for p in source_files)
        if len(source_files) + 2 > sdk.paths.MAX_MEMBERS:
            raise ValueError("App package exceeds the platform member limit")
        if payload_bytes + 1024 * 1024 > sdk.paths.MAX_UNPACKED_BYTES:
            raise ValueError("App package exceeds the platform unpacked limit")
        if (
            payload_bytes + environment_bytes
            > template["resources"]["limits"]["storage_mb"] * 1024 * 1024
        ):
            raise ValueError(
                "App payload plus expanded wheels exceeds its declared storage budget"
            )
        with contextlib.redirect_stdout(io.StringIO()):
            staged_package = Path(
                sdk.build(str(app), str(Path(temporary) / "packages"))
            )
        if staged_package.stat().st_size > sdk.paths.MAX_PKG_BYTES:
            raise ValueError("App package exceeds the platform compressed limit")
        out.mkdir(parents=True, exist_ok=True)
        package = out / staged_package.name
        # Publish only a fully validated output; failed checks preserve any
        # previous package in the caller's output directory.
        shutil.copyfile(staged_package, out / (package.name + ".tmp"))
        os.replace(out / (package.name + ".tmp"), package)
    record = {
        "file": package.name,
        "sha256": digest(package),
        "size": package.stat().st_size,
        "signed": False,
        "wheel_count": len(wheels),
        "expanded_environment_bytes": environment_bytes,
        "app_payload_bytes": payload_bytes,
        "validated_with": str(sdk_root / "market"),
        "validation": [
            "manifest-v2",
            "release-lock-bom",
            "wheel-dependency-closure",
            "offline-wheel-archive",
            "model-metadata-sha256",
        ],
        "source_wheel_sha256": next(
            w["sha256"]
            for w in wheel_audit["wheels"]
            if w["file"].startswith("inference_rv1126b-")
        ),
    }
    (out / (package.name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_app(
                args.sdk_root.resolve(),
                args.wheelhouse.resolve(),
                args.model_root.resolve(),
                args.out.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
