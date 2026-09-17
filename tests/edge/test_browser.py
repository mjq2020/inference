import os

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

from fastapi.testclient import TestClient

from inference.edge.api import create_app
from inference.edge.browser import BUILDER_ORIGIN
from inference.edge.settings import EdgeSettings
from tests.edge.test_api import FakeManager


def make_client(tmp_path):
    return TestClient(
        create_app(
            EdgeSettings(
                host="0.0.0.0", api_token="test-password", storage_root=tmp_path
            ),
            manager=FakeManager(),
        )
    )


def test_login_session_csrf_and_logout_revoke_online_editor(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/").status_code == 200
        assert client.get("/device", follow_redirects=False).status_code == 200
        assert client.get("/ui/assets/device.js").status_code == 200
        assert client.get("/app-center/workflow-runtime").status_code == 401
        assert client.get("/ui/session").status_code == 401
        assert client.post("/ui/login", json={"token": "wrong"}).status_code == 401
        response = client.post("/ui/login", json={"token": "test-password"})
        assert response.status_code == 200
        assert "HttpOnly" in response.headers["set-cookie"]
        session = response.json()
        assert "test-password" not in response.text
        assert client.get("/capabilities").status_code == 200
        assert client.post("/model/clear").status_code == 403
        csrf = {"X-CSRF": session["csrf"]}
        assert client.post("/model/clear", headers=csrf).status_code == 200
        response = client.post("/ui/builder-session", headers=csrf)
        grant = response.json()["csrf"]
        assert grant != session["csrf"]
        editor = {"Origin": BUILDER_ORIGIN, "X-CSRF": grant}
        # Deliberately omit cookies, as an embedded HTTPS editor must work
        # without third-party cookies on the device's HTTP listener.
        client.cookies.clear()
        response = client.get("/build/api", headers=editor)
        assert response.status_code == 200, response.text
        assert response.headers["access-control-allow-origin"] == BUILDER_ORIGIN
        assert client.get("/ui/session", headers=editor).status_code == 401
        assert client.post("/model/clear", headers=editor).status_code == 200
        assert (
            client.get(
                "/build/api", headers={**editor, "Origin": "https://evil.invalid"}
            ).status_code
            == 401
        )


def test_cookie_and_editor_grant_are_revoked_on_logout(tmp_path):
    with make_client(tmp_path) as client:
        session = client.post("/ui/login", json={"token": "test-password"}).json()
        csrf = {"X-CSRF": session["csrf"]}
        grant = client.post("/ui/builder-session", headers=csrf).json()["csrf"]
        assert client.post("/ui/logout", headers=csrf).status_code == 200
        assert client.get("/capabilities").status_code == 401
        assert (
            client.get(
                "/build/api", headers={"Origin": BUILDER_ORIGIN, "X-CSRF": grant}
            ).status_code
            == 401
        )


def test_cross_origin_login_and_cookie_writes_are_rejected(tmp_path):
    with make_client(tmp_path) as client:
        assert (
            client.post(
                "/ui/login",
                json={"token": "test-password"},
                headers={"Origin": "https://evil.invalid"},
            ).status_code
            == 403
        )
        session = client.post("/ui/login", json={"token": "test-password"}).json()
        assert (
            client.post(
                "/model/clear",
                headers={"X-CSRF": session["csrf"], "Origin": "https://evil.invalid"},
            ).status_code
            == 403
        )
        assert client.get("/ui/assets/../browser.py").status_code == 404


def test_login_failure_storage_is_bounded_and_rate_limited(tmp_path):
    with make_client(tmp_path) as client:
        for _ in range(10):
            assert client.post("/ui/login", json={"token": "wrong"}).status_code == 401
        assert client.post("/ui/login", json={"token": "wrong"}).status_code == 429


def test_editor_preflight_and_bearer_api_compatibility(tmp_path):
    with make_client(tmp_path) as client:
        response = client.options(
            "/workflows/run",
            headers={
                "Origin": BUILDER_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-csrf",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-private-network"] == "true"
        assert (
            client.post(
                "/model/clear", headers={"Authorization": "Bearer test-password"}
            ).status_code
            == 200
        )


def test_native_runtime_url_without_csrf_header_is_scoped_and_revocable(tmp_path):
    with make_client(tmp_path) as client:
        session = client.post("/ui/login", json={"token": "test-password"}).json()
        headers = {"X-CSRF": session["csrf"]}
        access = client.post("/ui/builder-session", headers=headers).json()
        assert client.post("/ui/builder-session", headers=headers).json() == access
        runtime = access["runtime_path"]
        cookie = client.cookies.get("inference_browser")
        client.cookies.clear()
        native = {"Origin": BUILDER_ORIGIN, "ngrok-skip-browser-warning": "true"}
        # The actual upstream runtime client omits X-CSRF, unlike Builder CRUD.
        response = client.get(runtime + "/model/registry", headers=native)
        assert response.status_code == 200, response.text
        assert response.headers["access-control-allow-origin"] == BUILDER_ORIGIN
        assert client.get(runtime + "/ui/session", headers=native).status_code == 401
        assert (
            client.get(
                runtime + "/model/registry", headers={"Origin": "https://evil.invalid"}
            ).status_code
            == 401
        )
        preflight = client.options(
            runtime + "/workflows/run",
            headers={
                **native,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "ngrok-skip-browser-warning,content-type",
            },
        )
        assert preflight.status_code == 200
        assert (
            "ngrok-skip-browser-warning"
            in preflight.headers["access-control-allow-headers"]
        )
        client.cookies.set("inference_browser", cookie)
        assert client.post("/ui/logout", headers=headers).status_code == 200
        client.cookies.clear()
        assert (
            client.get(runtime + "/model/registry", headers=native).status_code == 401
        )


def test_legacy_sdk_device_key_and_independent_cloud_key(tmp_path):
    with make_client(tmp_path) as client:
        assert (
            client.post("/model/clear", json={"api_key": "test-password"}).status_code
            == 200
        )
        assert client.get("/capabilities?api_key=test-password").status_code == 200
        assert client.post("/model/clear", json={"api_key": "wrong"}).status_code == 401

        @client.app.get("/test-key-boundary")
        async def key_boundary(request: __import__("fastapi").Request):
            return {"cloud": request.state.roboflow_api_key}

        response = client.get(
            "/test-key-boundary", headers={"Authorization": "Bearer test-password"}
        )
        assert response.json() == {"cloud": None}
        response = client.get(
            "/test-key-boundary",
            headers={
                "X-Inference-Token": "test-password",
                "Authorization": "Bearer cloud-test-key",
            },
        )
        assert response.json() == {"cloud": "cloud-test-key"}
        assert (
            client.get(
                "/test-key-boundary", headers={"Authorization": "Bearer cloud-test-key"}
            ).status_code
            == 401
        )


def test_canvas_session_survives_model_activation_and_password_rotation_revokes_it(tmp_path):
    with make_client(tmp_path) as client:
        session = client.post('/ui/login', json={'token': 'test-password'}).json()
        grant = client.post('/ui/builder-session', headers={'X-CSRF': session['csrf']}).json()['csrf']
        cookies = dict(client.cookies)
    with make_client(tmp_path) as restarted:
        restarted.cookies.update(cookies)
        assert restarted.get('/ui/session').status_code == 200
        assert restarted.get('/build/api', headers={'Origin': BUILDER_ORIGIN, 'X-CSRF': grant}).status_code == 200
        assert restarted.post('/ui/logout', headers={'X-CSRF': session['csrf']}).status_code == 200
    with make_client(tmp_path) as logged_out:
        assert logged_out.get('/build/api', headers={'Origin': BUILDER_ORIGIN, 'X-CSRF': grant}).status_code == 401
    from inference.edge.browser import BrowserAccess
    settings = EdgeSettings(api_token='first', storage_root=tmp_path)
    access = BrowserAccess(settings); owner, _ = access.new_session(); access.grants['grant'] = owner; access.persist()
    assert BrowserAccess(EdgeSettings(api_token='second', storage_root=tmp_path)).grants == {}
