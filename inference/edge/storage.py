"""Bounded local Workflow drafts, stored separately from immutable model releases."""

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from threading import RLock

from .errors import EdgeError
from .limits import encode_bounded_json

_FILE_NAME = re.compile(r"[0-9a-f]{64}\.json\Z")


def builder_document(document):
    """Expose legacy device drafts in the original Builder's config envelope.

    Existing original config strings are kept byte-for-byte, including canvas
    metadata unknown to this server. Legacy files are migrated in the response;
    the original file remains recoverable until the user saves in the Builder.
    """
    result = dict(document)
    if "config" not in result:
        specification = next(
            (
                result[key]
                for key in ("specification", "workflow", "definition")
                if key in result
            ),
            None,
        )
        if isinstance(specification, str):
            try:
                specification = json.loads(specification)
            except ValueError:
                return result  # Preserve incomplete drafts.
        if isinstance(specification, dict):
            config = {"specification": specification}
            if "edge_ui" in result:
                config["edge_ui"] = result["edge_ui"]
            result["config"] = json.dumps(
                config, ensure_ascii=False, separators=(",", ":")
            )
    return result


def workflow_specification(document):
    document = builder_document(document)
    config = document.get("config")
    try:
        if isinstance(config, str):
            config = json.loads(config)
        specification = config["specification"]
        if not isinstance(specification, dict):
            raise ValueError("specification must be an object")
    except (ValueError, TypeError, KeyError) as exc:
        raise EdgeError(
            "Saved Workflow has no valid specification",
            code="invalid_workflow",
            status_code=400,
        ) from exc
    return {**specification, "id": document.get("id")}


class WorkflowStore:
    """Preserve the original Builder config body, including unfinished drafts.

    Each write is atomic. A rename commits its destination before removing its
    source, so an interrupted save cannot destroy the previous Workflow.
    """

    def __init__(
        self,
        root,
        *,
        max_workflows=32,
        max_file_bytes=256 * 1024,
        max_total_bytes=768 * 1024,
    ):
        self.root = Path(root).absolute()
        self.max_workflows = max_workflows
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        if min(max_workflows, max_file_bytes, max_total_bytes) <= 0:
            raise ValueError("Workflow storage budgets must be positive")
        self._lock = RLock()
        self._ensure_root()

    @staticmethod
    def _identifier(identifier):
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[\w-]{1,128}", identifier)
            or identifier == "models"
        ):
            raise EdgeError(
                "Invalid or reserved Workflow identifier",
                code="invalid_workflow_id",
                status_code=400,
            )
        return identifier

    def _path(self, identifier):
        self._identifier(identifier)
        return self.root / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")

    def _ensure_root(self):
        if self.root.is_symlink():
            raise EdgeError(
                "Workflow data directory cannot be a symbolic link",
                code="workflow_storage_unavailable",
                status_code=503,
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise EdgeError(
                "Workflow data directory is unavailable",
                code="workflow_storage_unavailable",
                status_code=503,
            )

    @staticmethod
    def _regular_stat(path):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise EdgeError(
                "Workflow storage entry must be a regular file",
                code="workflow_storage_unavailable",
                status_code=503,
            )
        return metadata

    @staticmethod
    def _limit():
        raise EdgeError(
            "Workflow count or saved JSON size exceeds the device storage budget",
            code="workflow_storage_limit",
            status_code=413,
        )

    def _snapshot(self):
        entries, total = {}, 0
        with os.scandir(self.root) as directory:
            for item in directory:
                if not _FILE_NAME.fullmatch(item.name):
                    continue
                path = self.root / item.name
                metadata = self._regular_stat(path)
                if metadata is None:
                    continue
                if metadata.st_size > self.max_file_bytes:
                    self._limit()
                entries[item.name] = metadata
                total += metadata.st_size
                if len(entries) > self.max_workflows or total > self.max_total_bytes:
                    self._limit()
        return entries

    def _read(self, path):
        metadata = self._regular_stat(path)
        if metadata is None:
            raise EdgeError(
                "Workflow not found", code="workflow_not_found", status_code=404
            )
        if metadata.st_size > self.max_file_bytes:
            self._limit()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise EdgeError(
                    "Workflow changed while it was being read",
                    code="workflow_storage_unavailable",
                    status_code=503,
                )
            raw = handle.read(self.max_file_bytes + 1)
        if len(raw) > self.max_file_bytes:
            self._limit()
        try:
            config = json.loads(raw)
            if not isinstance(config, dict) or self._path(config.get("id")) != path:
                raise ValueError("Workflow identifier does not match its storage entry")
        except (ValueError, TypeError, RecursionError, EdgeError) as exc:
            raise EdgeError(
                "Saved Workflow data is invalid",
                code="workflow_storage_invalid",
                status_code=503,
            ) from exc
        return config, metadata

    def list(self):
        with self._lock:
            self._ensure_root()
            entries = self._snapshot()
            result = {}
            for name in sorted(entries):
                config, metadata = self._read(self.root / name)
                result[config["id"]] = {
                    "createTime": {"_seconds": int(metadata.st_ctime)},
                    "updateTime": {"_seconds": int(metadata.st_mtime)},
                    "config": config,
                }
            return {"data": result}

    def get(self, identifier):
        with self._lock:
            self._ensure_root()
            config, metadata = self._read(self._path(identifier))
            return {
                "data": {
                    "createTime": int(metadata.st_ctime),
                    "updateTime": int(metadata.st_mtime),
                    "config": config,
                }
            }

    def list_for_builder(self):
        result = self.list()
        for item in result["data"].values():
            item["config"] = builder_document(item["config"])
        return result

    def get_for_builder(self, identifier):
        result = self.get(identifier)
        result["data"]["config"] = builder_document(result["data"]["config"])
        return result

    def save(self, identifier, config):
        target = self._path(identifier)
        if not isinstance(config, dict):
            raise EdgeError(
                "Workflow config must be an object",
                code="invalid_workflow",
                status_code=400,
            )
        previous = config.get("id")
        old_path = self._path(previous) if previous and previous != identifier else None
        document = {**config, "id": identifier}
        try:
            payload = encode_bounded_json(document, self.max_file_bytes)
        except EdgeError:
            self._limit()
        except (ValueError, TypeError, RecursionError) as exc:
            raise EdgeError(
                "Workflow config must contain valid JSON",
                code="invalid_workflow",
                status_code=400,
            ) from exc
        with self._lock:
            self._ensure_root()
            entries = self._snapshot()
            if (
                old_path is not None
                and old_path.name in entries
                and target.name in entries
            ):
                raise EdgeError(
                    "A Workflow with the destination identifier already exists",
                    code="workflow_exists",
                    status_code=409,
                )
            removed = {target.name}
            if old_path is not None:
                removed.add(old_path.name)
            retained = {
                name: meta for name, meta in entries.items() if name not in removed
            }
            if (
                len(retained) + 1 > self.max_workflows
                or sum(meta.st_size for meta in retained.values()) + len(payload)
                > self.max_total_bytes
            ):
                self._limit()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.root,
                    prefix=".workflow-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = handle.name
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._regular_stat(target)
                os.replace(temporary, target)
                temporary = None
                if old_path is not None and old_path.name in entries:
                    old_path.unlink()
                directory = os.open(
                    self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
        return {"message": f"Workflow '{identifier}' created/updated successfully."}

    def delete(self, identifier):
        with self._lock:
            self._ensure_root()
            path = self._path(identifier)
            if self._regular_stat(path) is None:
                raise EdgeError(
                    "Workflow not found", code="workflow_not_found", status_code=404
                )
            path.unlink()
        return {"message": f"Workflow '{identifier}' deleted successfully."}
