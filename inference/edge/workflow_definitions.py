"""Local/cloud Workflow lookup without model loaders or cloud SDK dependencies."""

import copy
import hashlib
import json
import re
import time
from collections import OrderedDict
from threading import RLock
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .errors import EdgeError
from .storage import WorkflowStore, workflow_specification


class WorkflowDefinitions:
    def __init__(self, store, settings, *, fetcher=None):
        self.store, self.settings = store, settings
        self.fetcher = fetcher or self._fetch
        self._cache = OrderedDict()
        self._lock = RLock()
        self._disk = None

    def cloud_key(self, api_key=None):
        # The device credential authorizes local API access only. Never forward
        # it to Roboflow, including when an official SDK sends it in the body.
        if not api_key or api_key == self.settings.api_token:
            api_key = getattr(self.settings, "roboflow_api_key", None)
        if api_key == self.settings.api_token:
            return None
        return api_key or None

    def resolve(
        self,
        workspace,
        workflow_id,
        *,
        api_key=None,
        use_cache=True,
        workflow_version_id=None,
    ):
        if workspace == "local":
            if workflow_version_id:
                raise EdgeError(
                    "Local drafts do not have published version IDs",
                    code="unsupported_workflow_version",
                    status_code=400,
                )
            return workflow_specification(self.store.get(workflow_id)["data"]["config"])
        for value in (workspace, workflow_id):
            if not isinstance(value, str) or not re.fullmatch(r"[\w-]{1,128}", value):
                raise EdgeError(
                    "Invalid cloud Workflow identifier",
                    code="invalid_workflow_id",
                    status_code=400,
                )
        api_key = self.cloud_key(api_key)
        if api_key is None:
            raise EdgeError(
                "A separate Roboflow API key is required for cloud Workflows",
                code="missing_roboflow_api_key",
                status_code=400,
            )
        from inference.core.env import (
            USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS,
            WORKFLOWS_DEFINITION_CACHE_EXPIRY,
        )

        key = hashlib.sha256(
            json.dumps([workspace, workflow_id, workflow_version_id, api_key]).encode()
        ).hexdigest()
        with self._lock:
            entry = self._cache.get(key)
            if (
                use_cache
                and entry
                and time.time() - entry["fetched_at"]
                < WORKFLOWS_DEFINITION_CACHE_EXPIRY
            ):
                self._cache.move_to_end(key)
                return copy.deepcopy(entry["specification"])
            try:
                response = self.fetcher(
                    workspace, workflow_id, api_key, workflow_version_id
                )
                if not isinstance(response, dict) or not isinstance(
                    response.get("workflow"), dict
                ):
                    raise EdgeError(
                        "Roboflow returned an invalid Workflow response",
                        code="invalid_cloud_workflow",
                        status_code=502,
                    )
                specification = workflow_specification(response["workflow"])
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                # Match the upstream network-failure fallback, never substitute
                # a cached definition after an authorization/HTTP error.
                if use_cache and USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
                    cached = entry or self._load_disk(key)
                    if cached:
                        return copy.deepcopy(cached["specification"])
                raise EdgeError(
                    "Cannot retrieve the cloud Workflow",
                    code="cloud_workflow_unavailable",
                    status_code=503,
                ) from exc
            entry = {"fetched_at": time.time(), "specification": specification}
            if use_cache:
                self._cache[key] = entry
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
                if USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
                    self._save_disk(key, entry)
            return copy.deepcopy(specification)

    def _disk_store(self):
        if self._disk is None:
            self._disk = WorkflowStore(
                self.settings.storage_root / "cloud-workflows",
                max_workflows=4,
                max_file_bytes=1024 * 1024,
                max_total_bytes=4 * 1024 * 1024,
            )
        return self._disk

    def _load_disk(self, key):
        try:
            return self._disk_store().get(key)["data"]["config"]
        except (EdgeError, OSError):
            return None

    def _save_disk(self, key, entry):
        disk = self._disk_store()
        current = disk.list()["data"]
        if key not in current and len(current) >= 4:
            oldest = min(
                current, key=lambda k: current[k]["config"].get("fetched_at", 0)
            )
            disk.delete(oldest)
        disk.save(key, entry)

    @staticmethod
    def _fetch(workspace, workflow_id, api_key, workflow_version_id):
        from inference.core.env import API_BASE_URL

        params = {"api_key": api_key}
        if workflow_version_id is not None:
            params["workflow_version"] = workflow_version_id
        url = f"{API_BASE_URL.rstrip('/')}/{quote(workspace, safe='')}/workflows/{quote(workflow_id, safe='')}?{urlencode(params)}"
        try:
            with urlopen(
                Request(url, headers={"Accept": "application/json"}), timeout=15
            ) as response:
                raw = response.read(1024 * 1024 + 1)
        except HTTPError as exc:
            status = exc.code if exc.code in (401, 403, 404, 429) else 502
            raise EdgeError(
                "Roboflow rejected the Workflow request",
                code="cloud_workflow_http_error",
                status_code=status,
            ) from exc
        if len(raw) > 1024 * 1024:
            raise EdgeError(
                "Cloud Workflow exceeds the definition size budget",
                code="workflow_too_large",
                status_code=413,
            )
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise EdgeError(
                "Roboflow returned invalid JSON",
                code="invalid_cloud_workflow",
                status_code=502,
            ) from exc
