"""Small browser sessions and a narrowly scoped grant for the online Builder."""

import hmac
import hashlib
import json
import os
import re
import secrets
import time
from collections import OrderedDict
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .errors import EdgeError

BUILDER_ORIGIN = "https://app.roboflow.com"
COOKIE = "inference_browser"
SESSION_SECONDS = 12 * 60 * 60
STATIC = Path(__file__).parent / "static"


class BrowserAccess:
    def __init__(self, settings):
        self.settings = settings
        self.sessions = OrderedDict()
        self.grants = OrderedDict()
        self.failures = []
        self._restore()

    def _session_file(self):
        return self.settings.storage_root / ".browser-sessions.json"

    def _fingerprint(self):
        return hashlib.sha256(self.settings.api_token.encode()).hexdigest()

    def _restore(self):
        try:
            descriptor = os.open(self._session_file(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                value = json.loads(stream.read(65537))
            if value.get("credential") != self._fingerprint():
                return
            now, wall = time.monotonic(), time.time()
            for key, item in list(value.get("sessions", {}).items())[-32:]:
                remaining = min(SESSION_SECONDS, item["expires"] - wall)
                if remaining > 0 and isinstance(item.get("csrf"), str):
                    self.sessions[key] = {"csrf": item["csrf"], "expires": now + remaining}
            for key, owner in list(value.get("grants", {}).items())[-32:]:
                if owner in self.sessions:
                    self.grants[key] = owner
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            self.sessions.clear()
            self.grants.clear()

    def persist(self):
        """Keep existing canvas grants across model activation; token changes revoke them."""
        self.prune()
        path = self._session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        now, wall = time.monotonic(), time.time()
        value = {"credential": self._fingerprint(), "grants": dict(self.grants),
                 "sessions": {key: {**item, "expires": wall + item["expires"] - now}
                              for key, item in self.sessions.items()}}
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def prune(self):
        now = time.monotonic()
        for key, value in list(self.sessions.items()):
            if value["expires"] <= now:
                del self.sessions[key]
        for key, session_id in list(self.grants.items()):
            if session_id not in self.sessions:
                del self.grants[key]

    def session(self, headers):
        self.prune()
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get(b"cookie", b"").decode("latin-1"))
            session_id = cookie[COOKIE].value if COOKIE in cookie else ""
        except Exception:
            return None, None
        return session_id, self.sessions.get(session_id)

    def new_session(self):
        self.prune()
        while len(self.sessions) >= 32:
            self.sessions.popitem(last=False)
        key = secrets.token_urlsafe(32)
        session = {
            "csrf": secrets.token_urlsafe(32),
            "expires": time.monotonic() + SESSION_SECONDS,
        }
        self.sessions[key] = session
        self.persist()
        return key, session

    def payload(self, session):
        settings = self.settings
        return {
            "authenticated": True,
            "csrf": session["csrf"],
            "settings": {
                "host": settings.host,
                "port": settings.port,
                "max_fps": settings.max_fps,
                "max_workflow_steps": settings.max_workflow_steps,
                "max_request_bytes": settings.max_request_bytes,
                "max_response_bytes": settings.max_response_bytes,
            },
        }

    @staticmethod
    def editor_path(path):
        return (
            path == "/build/api"
            or path.startswith("/build/api/")
            or path
            in {
                "/workflows/execution_engine/versions",
                "/workflows/blocks/describe",
                "/workflows/definition/schema",
                "/workflows/blocks/dynamic_outputs",
                "/workflows/validate",
                "/workflows/run",
                "/workflows/describe_interface",
                "/infer/workflows",
                "/model/registry",
                "/model/add",
                "/model/remove",
                "/model/clear",
                "/infer/object_detection",
                "/initialise_webrtc_worker",
                "/capabilities",
                "/healthz",
                "/info",
            }
            or path.startswith("/inference_pipelines/")
            or path.startswith("/webrtc/")
            or bool(
                re.fullmatch(r"/(?:infer/workflows/)?[\w-]+/workflows/[\w-]+", path)
            )
            or bool(re.fullmatch(r"/infer/workflows/[\w-]+/[\w-]+", path))
        )


class BrowserAuthMiddleware:
    def __init__(self, app, access):
        self.app, self.access = app, access

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        access = self.access
        headers = dict(scope.get("headers", []))
        path, method = scope["path"], scope["method"]
        capability = None
        if path.startswith("/ui/runtime/"):
            parts = path.split("/", 4)
            if len(parts) == 5:
                capability, path = parts[3], "/" + parts[4]
            else:
                capability = ""
        origin = headers.get(b"origin", b"").decode("latin-1")
        host = headers.get(b"host", b"").decode("latin-1")
        same_origin = not origin or origin == f'{scope.get("scheme", "http")}://{host}'
        editor = origin == BUILDER_ORIGIN and access.editor_path(path)
        cors = (
            {
                "Access-Control-Allow-Origin": BUILDER_ORIGIN,
                "Vary": "Origin",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": (
                    "Content-Type, Accept, X-CSRF, Authorization, "
                    "X-Inference-Token, X-Roboflow-Api-Key, X-API-Key, "
                    "ngrok-skip-browser-warning"
                ),
                "Access-Control-Allow-Private-Network": "true",
            }
            if editor
            else {}
        )

        async def respond(message, status):
            return await JSONResponse({"error": message}, status, headers=cors)(
                scope, receive, send
            )

        public = path in (
            "/",
            "/ui/",
            "/build",
            "/build/",
            "/device",  # Public bookmark redirect; no runtime data is exposed.
            "/ui/login",
            "/healthz",
            "/info",
            "/favicon.ico",
        ) or path.startswith(("/ui/assets/", "/build/edit/"))
        # These schemas describe installed code only, never saved workflows,
        # credentials or model files. The native runtime discovers them before
        # its connection API key is configured.
        metadata_request = (
            method in ("GET", "HEAD")
            and path in {"/workflows/execution_engine/versions", "/workflows/definition/schema", "/workflows/blocks/describe"}
        ) or (method == "POST" and path == "/workflows/blocks/describe")
        public = public or metadata_request
        session_id, session = access.session(headers)
        authorization = headers.get(b"authorization", b"")
        header_key = (
            authorization[7:] if authorization[:7].lower() == b"bearer " else b""
        )
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        legacy_key = scope.get("state", {}).get("legacy_api_key", "")
        candidates = [header_key, headers.get(b"x-inference-token", b"")]
        candidates += [value.encode() for value in query.get("api_key", [])[:1]]
        if isinstance(legacy_key, str):
            candidates.append(legacy_key.encode())
        bearer = bool(access.settings.api_token) and any(
            hmac.compare_digest(value, access.settings.api_token.encode())
            for value in candidates
        )
        csrf = headers.get(b"x-csrf", b"").decode("latin-1")
        grant_session = access.grants.get(
            capability if capability is not None else csrf
        )
        grant = bool(grant_session and grant_session in access.sessions)
        grant = (
            grant
            and access.editor_path(path)
            and (editor or (capability is not None and same_origin))
        )
        if capability is not None and not grant:
            return await respond("unauthorized", 401)
        if method == "OPTIONS" and editor:
            return await JSONResponse({}, 200, headers=cors)(scope, receive, send)
        if (
            not public
            and not bearer
            and not grant
            and not session
            and access.settings.api_token
        ):
            if method == "GET" and (
                path in {"/build", "/device", "/legacy"} or path.startswith("/build/edit/")
            ):
                return await RedirectResponse("/", status_code=303)(
                    scope, receive, send
                )
            return await respond("unauthorized", 401)
        if not same_origin and not grant and not (editor and bearer):
            # Public static files contain no credentials. All browser API access
            # must be same-origin or carry an explicit, revocable editor grant.
            if not (public and method in ("GET", "HEAD")) and not metadata_request:
                return await respond("origin_not_allowed", 403)
        if (
            session
            and not bearer
            and not grant
            and not metadata_request
            and method not in ("GET", "HEAD", "OPTIONS")
        ):
            if not hmac.compare_digest(csrf.encode(), session["csrf"].encode()):
                return await respond("invalid_csrf", 403)

        async def secure_send(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    (k.lower().encode(), v.encode()) for k, v in cors.items()
                )
                response_headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message = {**message, "headers": response_headers}
            await send(message)

        scope.setdefault("state", {})["browser_session"] = session
        scope["state"]["browser_session_id"] = session_id
        cloud_key = headers.get(b"x-roboflow-api-key", b"") or header_key
        try:
            cloud_key_text = cloud_key.decode("utf-8")
        except UnicodeDecodeError:
            return await respond("invalid_api_key_encoding", 400)
        scope["state"]["roboflow_api_key"] = (
            cloud_key_text
            if cloud_key
            and not hmac.compare_digest(cloud_key, access.settings.api_token.encode())
            else None
        )
        if capability is not None:
            # The original Builder authenticates CRUD with X-CSRF, but its
            # runtime client does not propagate that header. A revocable
            # session URL preserves that client without exposing the durable
            # device password or trusting Origin alone.
            scope = {**scope, "path": path, "raw_path": path.encode()}
        await self.app(scope, receive, secure_send)


def install_browser(app, settings):
    access = BrowserAccess(settings)
    app.state.browser_access = access
    app.add_middleware(BrowserAuthMiddleware, access=access)
    router = APIRouter()
    server_id = str(uuid4())

    @router.get("/info")
    async def server_info():
        from inference.core.version import __version__

        return {
            "name": "Roboflow Inference Server",
            "version": __version__,
            "uuid": server_id,
            "device_app_version": "0.3.1",
            "runtime_profile": "rv1126b",
        }

    def session_response(request, session_id, session):
        response = JSONResponse(access.payload(session))
        response.set_cookie(
            COOKIE,
            session_id,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=SESSION_SECONDS,
        )
        return response

    @router.get("/ui/session")
    async def session_info(request: Request):
        session = request.state.browser_session
        if session:
            return access.payload(session)
        # A loopback-only deployment can be used without setting a password.
        # On an exposed listener the outer middleware has already required a
        # valid API bearer; it may exchange that bearer for a browser session.
        session_id, session = access.new_session()
        return session_response(request, session_id, session)

    @router.post("/ui/login")
    async def login(request: Request):
        now = time.monotonic()
        access.failures[:] = [value for value in access.failures if value > now - 60]
        if len(access.failures) >= 10:
            raise EdgeError(
                "Please retry after one minute",
                code="login_rate_limited",
                status_code=429,
            )
        try:
            body = await request.json()
        except ValueError:
            body = None
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not hmac.compare_digest(
            token.encode(), settings.api_token.encode()
        ):
            access.failures.append(now)
            raise EdgeError(
                "Invalid access password", code="unauthorized", status_code=401
            )
        session_id, session = access.new_session()
        return session_response(request, session_id, session)

    @router.post("/ui/logout")
    async def logout(request: Request):
        access.sessions.pop(request.state.browser_session_id, None)
        access.prune()
        access.persist()
        response = JSONResponse({"status": "ok"})
        response.delete_cookie(COOKIE)
        return response

    @router.post("/ui/builder-session")
    async def builder_session(request: Request):
        session_id = request.state.browser_session_id
        if session_id not in access.sessions:
            raise EdgeError(
                "Log in to open the online Builder",
                code="unauthorized",
                status_code=401,
            )
        access.prune()
        # Reopening the entry must not invalidate another canvas tab.
        token = next(
            (key for key, owner in access.grants.items() if owner == session_id), None
        )
        if token is None:
            token = secrets.token_urlsafe(32)
            access.grants[token] = session_id
            access.persist()
        return {
            "origin": BUILDER_ORIGIN,
            "csrf": token,
            "runtime_path": "/ui/runtime/" + token,
        }

    @router.get("/")
    @router.get("/ui/")
    async def home():
        return FileResponse(
            STATIC / "build.html",
            headers={"Content-Security-Policy": "frame-ancestors 'self'"},
        )

    @router.get("/build")
    @router.get("/build/edit/{workflow_id}")
    async def builder_page():
        return FileResponse(
            STATIC / "build.html",
            headers={"Content-Security-Policy": "frame-ancestors 'self'"},
        )

    @router.get("/legacy")
    async def legacy_page():
        return FileResponse(
            STATIC / "index.html",
            headers={"Content-Security-Policy": "frame-ancestors 'self'"},
        )

    @router.get("/device")
    async def device_page():
        return FileResponse(
            STATIC / "device.html",
            headers={"Content-Security-Policy": "frame-ancestors 'self'"},
        )

    @router.get("/build/")
    async def builder_redirect():
        return RedirectResponse("/build", status_code=302)

    @router.get("/favicon.ico")
    async def favicon():
        return JSONResponse({}, status_code=404)

    app.include_router(router)
    app.mount(
        "/ui/assets", StaticFiles(directory=STATIC, check_dir=False), name="edge-assets"
    )
    return access
