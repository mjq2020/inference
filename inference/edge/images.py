"""Bounded uploaded-image decoding. The internal array contract is BGR uint8."""

import base64
import binascii
import io
import time
from urllib.parse import urljoin, urlsplit

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import EdgeError


def _http_url(value):
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme.lower() in ("http", "https")
            and parsed.hostname
            and parsed.port != 0
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise EdgeError(
            "Image URLs must use HTTP or HTTPS",
            code="unsupported_image_url",
            status_code=422,
        )
    return value


def _download_image(url, *, max_bytes, timeout, total_timeout):
    import requests
    from urllib3.exceptions import ReadTimeoutError

    deadline = time.monotonic() + total_timeout
    url = _http_url(url)
    try:
        # A separate session never receives the caller's API token or cookies.
        with requests.Session() as session:
            for redirect in range(4):
                if time.monotonic() > deadline:
                    raise requests.Timeout()
                with session.get(
                    url,
                    stream=True,
                    allow_redirects=False,
                    timeout=timeout,
                    headers={"Accept": "image/*", "Accept-Encoding": "gzip, deflate"},
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirect == 3 or not response.headers.get("Location"):
                            raise EdgeError(
                                "Image URL exceeded the redirect limit",
                                code="image_download_failed",
                                status_code=502,
                            )
                        url = _http_url(urljoin(url, response.headers["Location"]))
                        continue
                    response.raise_for_status()
                    try:
                        length = int(response.headers.get("Content-Length", "0"))
                    except ValueError:
                        length = 0
                    if length > max_bytes:
                        raise EdgeError(
                            "Downloaded image exceeds byte budget",
                            code="image_too_large",
                            status_code=413,
                        )
                    chunks, size = [], 0
                    # read1 returns after one buffered read, allowing the total
                    # deadline to stop peers that continuously trickle bytes.
                    while True:
                        if time.monotonic() > deadline:
                            raise requests.Timeout()
                        part = response.raw.read1(
                            min(65536, max_bytes - size + 1), decode_content=True
                        )
                        if not part:
                            break
                        size += len(part)
                        if size > max_bytes:
                            raise EdgeError(
                                "Downloaded image exceeds byte budget",
                                code="image_too_large",
                                status_code=413,
                            )
                        chunks.append(part)
                    return b"".join(chunks)
    except EdgeError:
        raise
    except (requests.Timeout, ReadTimeoutError) as exc:
        raise EdgeError(
            "Image download timed out", code="image_download_timeout", status_code=504
        ) from exc
    except Exception as exc:
        # Do not return source credentials, URLs, response bodies or proxy
        # configuration embedded in third-party exception strings.
        raise EdgeError(
            "Cannot download image URL", code="image_download_failed", status_code=502
        ) from exc


def decode_image(
    value,
    *,
    max_pixels=4096 * 2160,
    max_bytes=8 * 1024 * 1024,
    timeout=(3, 5),
    total_timeout=20
):
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 3
            or value.shape[2] != 3
            or value.dtype != np.uint8
            or not value.size
        ):
            raise EdgeError(
                "Expected HWC uint8 BGR image", code="invalid_image", status_code=422
            )
        if value.shape[0] * value.shape[1] > max_pixels:
            raise EdgeError(
                "Image exceeds pixel budget", code="image_too_large", status_code=413
            )
        return value
    if isinstance(value, dict):
        kind = value.get("type")
        if kind not in ("base64", "numpy", "url"):
            raise EdgeError(
                "Use an HTTP(S) URL, base64 upload or NumPy image",
                code="unsupported_image_type",
                status_code=422,
            )
        payload = value.get("value")
        if (kind in ("base64", "url") and not isinstance(payload, str)) or (
            kind == "numpy" and not isinstance(payload, np.ndarray)
        ):
            raise EdgeError(
                "Image value does not match its type",
                code="invalid_image",
                status_code=422,
            )
        if kind == "url":
            raw = _download_image(
                payload,
                max_bytes=max_bytes,
                timeout=timeout,
                total_timeout=total_timeout,
            )
            return _decode_bytes(raw, max_pixels=max_pixels)
        if kind == "numpy":
            return decode_image(payload, max_pixels=max_pixels, max_bytes=max_bytes)
        value = payload
    elif isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        raw = _download_image(
            value, max_bytes=max_bytes, timeout=timeout, total_timeout=total_timeout
        )
        return _decode_bytes(raw, max_pixels=max_pixels)
    if not isinstance(value, str):
        raise EdgeError(
            "Expected a base64 image", code="invalid_image", status_code=422
        )
    if value.startswith("data:image/"):
        if ";base64," not in value:
            raise EdgeError(
                "Expected a base64 data URI", code="invalid_image", status_code=422
            )
        value = value.split(";base64,", 1)[1]
    try:
        if len(value) > 4 * ((max_bytes + 2) // 3):
            raise EdgeError(
                "Encoded image exceeds byte budget",
                code="image_too_large",
                status_code=413,
            )
        raw = base64.b64decode(value, validate=True)
        if len(raw) > max_bytes:
            raise EdgeError(
                "Encoded image exceeds byte budget",
                code="image_too_large",
                status_code=413,
            )
    except (ValueError, binascii.Error) as exc:
        raise EdgeError(
            "Invalid encoded image", code="invalid_image", status_code=422
        ) from exc
    return _decode_bytes(raw, max_pixels=max_pixels)


def _decode_bytes(raw, *, max_pixels):
    try:
        with Image.open(io.BytesIO(raw)) as header:
            width, height = header.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise EdgeError(
                    "Image exceeds pixel budget",
                    code="image_too_large",
                    status_code=413,
                )
        array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (
        ValueError,
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise EdgeError(
            "Invalid encoded image", code="invalid_image", status_code=422
        ) from exc
    if array is None or array.shape[0] * array.shape[1] > max_pixels:
        raise EdgeError(
            "Invalid or oversized image", code="invalid_image", status_code=422
        )
    return array
