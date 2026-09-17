import base64
import gzip
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"

import cv2
import numpy as np
import pytest

from inference.edge.errors import EdgeError
from inference.edge.images import decode_image
from inference.edge.webrtc import preview_image


@pytest.fixture
def image_server():
    image = np.zeros((16, 24, 3), np.uint8)
    image[:] = [20, 40, 60]
    _, encoded = cv2.imencode(".png", image)
    raw = encoded.tobytes()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            try:
                if self.path == "/slow":
                    time.sleep(0.2)
                if self.path in ("/redirect", "/loop", "/file"):
                    self.send_response(302)
                    self.send_header(
                        "Location",
                        {
                            "/redirect": "/image",
                            "/loop": "/loop",
                            "/file": "file:///etc/passwd",
                        }[self.path],
                    )
                    self.end_headers()
                    return
                self.send_response(200)
                if self.path == "/large-header":
                    self.send_header("Content-Length", "1000000000")
                if self.path == "/gzip":
                    self.send_header("Content-Encoding", "gzip")
                self.end_headers()
                if self.path == "/stream-large":
                    self.wfile.write(b"x" * 2048)
                elif self.path == "/trickle":
                    for _ in range(30):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.01)
                elif self.path == "/gzip":
                    self.wfile.write(gzip.compress(raw))
                elif self.path != "/large-header":
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", raw, image, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_standard_url_redirect_gzip_and_existing_image_inputs(image_server):
    url, raw, expected, received = image_server
    for suffix in ("/image", "/redirect", "/gzip"):
        actual = decode_image({"type": "url", "value": url + suffix})
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(decode_image(url + "/image"), expected)
    np.testing.assert_array_equal(
        decode_image({"type": "base64", "value": base64.b64encode(raw).decode()}),
        expected,
    )
    assert decode_image(expected) is expected
    assert all(
        "Authorization" not in headers and "Cookie" not in headers
        for _, headers in received
    )


@pytest.mark.parametrize("suffix", ["/large-header", "/stream-large"])
def test_url_download_byte_limit_precedes_decode(image_server, monkeypatch, suffix):
    url, _, _, _ = image_server
    monkeypatch.setattr(
        cv2, "imdecode", lambda *a: pytest.fail("oversized response reached decoding")
    )
    with pytest.raises(EdgeError) as exc:
        decode_image({"type": "url", "value": url + suffix}, max_bytes=1024)
    assert exc.value.status_code == 413


def test_url_pixel_limit_and_disallowed_redirects(image_server, monkeypatch):
    url, _, _, _ = image_server
    monkeypatch.setattr(
        cv2, "imdecode", lambda *a: pytest.fail("oversized pixels reached decoding")
    )
    with pytest.raises(EdgeError) as exc:
        decode_image({"type": "url", "value": url + "/image"}, max_pixels=100)
    assert exc.value.status_code == 413
    for value in ("file:///etc/passwd", "ftp://localhost/test", url + "/file"):
        with pytest.raises(EdgeError) as exc:
            decode_image({"type": "url", "value": value})
        assert exc.value.status_code == 422
    with pytest.raises(EdgeError) as exc:
        decode_image({"type": "url", "value": url + "/loop"})
    assert exc.value.status_code == 502


@pytest.mark.parametrize("suffix,total", [("/slow", 1), ("/trickle", 0.04)])
def test_url_connection_read_and_total_time_limits(image_server, suffix, total):
    url, _, _, _ = image_server
    with pytest.raises(EdgeError) as exc:
        decode_image(
            {"type": "url", "value": url + suffix},
            timeout=(0.05, 0.05),
            total_timeout=total,
        )
    assert exc.value.status_code == 504


def test_preview_resize_preserves_input_and_original_coordinate_contract():
    image = np.zeros((2160, 3840, 3), np.uint8)
    image[0, 0] = [10, 20, 30]
    prediction = {
        "image": {"width": 3840, "height": 2160},
        "predictions": [{"x": 1920, "y": 1080, "width": 100, "height": 200}],
    }
    resized = preview_image(image)
    assert resized.shape == (720, 1280, 3)
    assert image.shape == (2160, 3840, 3)
    assert image[0, 0].tolist() == [10, 20, 30]
    assert prediction["predictions"][0]["x"] == 1920
    assert prediction["image"]["width"] == 3840
    assert preview_image(resized) is resized
