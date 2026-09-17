"""Derived wheel correctness, provenance and firmware dependency boundaries."""

import base64
import csv
import hashlib
import importlib
import io
import json
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "deploy/rv1126b"))
    return importlib.import_module("build_wheel"), importlib.import_module(
        "audit_wheels"
    )


def verify_record(path):
    with zipfile.ZipFile(path) as archive:
        record = next(name for name in archive.namelist() if name.endswith("/RECORD"))
        for name, checksum, size in csv.reader(
            io.StringIO(archive.read(record).decode())
        ):
            if name == record:
                continue
            content = archive.read(name)
            assert int(size) == len(content)
            expected = base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).rstrip(b"=")
            assert checksum == "sha256=" + expected.decode()


def fake_wheel(
    build,
    directory,
    *,
    name="example",
    version="4.65.0",
    tag="py2.py3-none-any",
    files=None,
):
    dist_info = f"{name}-{version}.dist-info"
    content = {
        dist_info
        + "/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode(),
        dist_info
        + "/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
        name + "/__init__.py": b"VALUE = 42\n",
    }
    content.update(files or {})
    return build.write_wheel(
        directory / f"{name}-{version}-{tag}.whl", content, dist_info
    )


@pytest.fixture
def font_assets(tmp_path, tools, monkeypatch):
    """Exercise the complete real registry without network or binary test fixtures."""
    build, _ = tools
    directory = tmp_path / "fonts"
    registry = {}
    for identifier, metadata in build.font_registry().items():
        font = (identifier + " font fixture\n").encode()
        license_text = (identifier + " license fixture\n").encode()
        target = directory / identifier
        target.mkdir(parents=True)
        (target / metadata.file_name).write_bytes(font)
        (target / "OFL.txt").write_bytes(license_text)
        registry[identifier] = SimpleNamespace(
            file_name=metadata.file_name,
            sha256=build.digest(font),
            license_sha256=build.digest(license_text),
            source_url=metadata.source_url,
            license_url=metadata.license_url,
        )
    monkeypatch.setattr(build, "font_registry", lambda: registry)
    return directory


def test_source_wheel_determinism_and_model_stack_exclusion(
    tmp_path, tools, font_assets
):
    build, _ = tools
    first = build.build_source_wheel(tmp_path / "first", fonts_dir=font_assets)
    second = build.build_source_wheel(tmp_path / "second", fonts_dir=font_assets)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert "inference_edge_probe.py" in names
        assert "inference/core/workflows/execution_engine/core.py" in names
        assert not any(name.startswith(build.FORBIDDEN_PATHS) for name in names)
        source_manifest = json.loads(
            archive.read(
                next(name for name in names if name.endswith("/SOURCE_MANIFEST.json"))
            )
        )
        for asset in (
            "index.html",
            "build.html",
            "app.css",
            "common.js",
            "app.js",
            "builder.js",
            "device.html",
            "device.js",
        ):
            name = "inference/edge/static/" + asset
            assert name in names
            assert (
                source_manifest[name] == hashlib.sha256(archive.read(name)).hexdigest()
            )
        catalog = runpy.run_path(
            str(ROOT / "inference/core/workflows/core_steps/catalog_rv1126b.py")
        )["BLOCK_MODULES"]
        assert len(catalog) > 100
        for module in catalog:
            assert module.replace(".", "/") + ".py" in names
        for helper in (
            "inputs_discovery",
            "outputs_discovery",
            "types_discovery",
            "kinds_schemas",
            "kinds_schemas_register",
        ):
            assert (
                "inference/core/workflows/execution_engine/v1/introspection/"
                + helper
                + ".py"
            ) in names
        provenance = json.loads(
            archive.read(
                next(name for name in names if name.endswith("/FONT_ASSETS.json"))
            )
        )
        assert len(provenance) == 2 * len(build.font_registry()) == 40
        for name, origin in provenance.items():
            assert (
                source_manifest[name]
                == origin["sha256"]
                == build.digest(archive.read(name))
            )
            assert origin["source_url"].startswith("https://")
        for name, checksum in source_manifest.items():
            assert checksum == build.digest(archive.read(name))
    verify_record(first)


@pytest.mark.parametrize("failure", ["missing-font", "license-checksum", "symlink"])
def test_offline_build_rejects_incomplete_or_unverified_fonts(
    tmp_path, tools, font_assets, failure
):
    build, _ = tools
    identifier, metadata = next(iter(build.font_registry().items()))
    font = font_assets / identifier / metadata.file_name
    if failure == "missing-font":
        font.unlink()
    elif failure == "license-checksum":
        (font.parent / "OFL.txt").write_text("unverified replacement")
    else:
        outside = tmp_path / "outside-font"
        font.rename(outside)
        font.symlink_to(outside)
    with pytest.raises(ValueError, match="font asset"):
        build.collect_font_assets(font_assets)


def test_python_tag_normalization_preserves_sources_and_provenance(tmp_path, tools):
    build, audit = tools
    source = fake_wheel(build, tmp_path)
    original_hash = build.digest(source.read_bytes())
    derived = audit.normalize(source)
    assert derived.name.endswith("-py3-none-any.whl")
    assert not source.exists()
    with zipfile.ZipFile(derived) as archive:
        assert archive.read("example/__init__.py") == b"VALUE = 42\n"
        provenance = json.loads(
            archive.read("example-4.65.0.dist-info/RV1126B_WHEEL_ADAPTATION.json")
        )
        assert provenance["source_sha256"] == original_hash
        assert not provenance["binary_abi_changed"]
    verify_record(derived)


def test_only_known_nonruntime_fonttools_manpage_is_removed(tmp_path, tools):
    build, audit = tools
    manual = "fonttools-4.65.0.data/data/share/man/man1/ttx.1"
    source = fake_wheel(build, tmp_path, name="fonttools", files={manual: b"manual"})
    derived = audit.normalize(source)
    with zipfile.ZipFile(derived) as archive:
        assert manual not in archive.namelist()
        assert archive.read("fonttools/__init__.py") == b"VALUE = 42\n"
    verify_record(derived)
    other = fake_wheel(
        build,
        tmp_path / "other",
        files={"example-4.65.0.data/data/model.bin": b"model"},
    )
    with pytest.raises(ValueError, match=".data"):
        audit.normalize(other)


def test_stable_abi_can_narrow_to_cp311_but_newer_abis_cannot(tmp_path, tools):
    build, audit = tools
    tag = "cp310-abi3-manylinux_2_28_aarch64"
    binary = b"unchanged stable ABI binary fixture"
    source = fake_wheel(
        build, tmp_path, tag=tag, files={"example/native.abi3.so": binary}
    )
    derived = audit.normalize(source)
    assert "cp311-abi3" in derived.name
    with zipfile.ZipFile(derived) as archive:
        assert archive.read("example/native.abi3.so") == binary
        provenance = json.loads(
            archive.read("example-4.65.0.dist-info/RV1126B_WHEEL_ADAPTATION.json")
        )
        assert provenance["python_compatibility_narrowed"]
        assert not provenance["binary_abi_changed"]
    verify_record(derived)
    for tag in (
        "cp312-abi3-manylinux_2_28_aarch64",
        "cp310-cp310-manylinux_2_28_aarch64",
    ):
        source = fake_wheel(build, tmp_path / tag, tag=tag)
        with pytest.raises(ValueError, match="wheel tag"):
            audit.normalize(source)


def test_system_library_override_fails_audit(tmp_path, tools):
    build, audit = tools
    fake_wheel(build, tmp_path, tag="py3-none-any", files={"cv2/__init__.py": b""})
    with pytest.raises(ValueError, match="override"):
        audit.audit(tmp_path)


def test_networkx_patch_defers_bz2_until_compressed_io(tmp_path, tools, monkeypatch):
    build, _ = tools
    source_code = b"""import bz2
fopeners = {
    ".bz2": bz2.BZ2File,
}
"""
    source = fake_wheel(
        build,
        tmp_path,
        name="networkx",
        version="3.4.2",
        tag="py3-none-any",
        files={"networkx/utils/decorators.py": source_code},
    )
    derived = build.build_networkx_wheel(source, tmp_path / "derived")
    with zipfile.ZipFile(derived) as archive:
        patched = archive.read("networkx/utils/decorators.py").decode()
        assert archive.read("networkx/__init__.py") == b"VALUE = 42\n"
    namespace = {}
    exec(compile(patched, "networkx-decorators-test", "exec"), namespace)
    # Existing bz2 is not replaced in production; only this test simulates its absence.
    monkeypatch.setitem(sys.modules, "bz2", None)
    target = tmp_path / "unexpected.bz2"
    with pytest.raises(RuntimeError, match="no _bz2"):
        namespace["fopeners"][".bz2"](target, "wb")
    assert not target.exists()
    verify_record(derived)
