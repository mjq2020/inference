"""Per-execution budgets shared by NumPy Workflow steps and serializers."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from .errors import EdgeError

_CURRENT = ContextVar("rv1126b_workflow_budget", default=None)


def exceeded(message):
    raise EdgeError(message, code="resource_budget", status_code=422)


@dataclass
class WorkflowBudget:
    max_pixels: int
    max_json_bytes: int
    allow_images: bool = True
    max_crops: int = 32
    crops: int = 0
    crop_pixels: int = 0
    model_calls: int = 0
    operations: int = 0
    output_pixels: int = 0


@contextmanager
def workflow_budget(settings, *, video=False):
    token = _CURRENT.set(
        WorkflowBudget(
            max_pixels=settings.max_image_pixels,
            max_json_bytes=settings.max_response_bytes,
            # Video uses the same cumulative image/JSON limits as still images.
            # The video runner retains only two serialized results, never frames.
            allow_images=True,
        )
    )
    try:
        yield
    finally:
        _CURRENT.reset(token)


def reserve_crop(pixels):
    budget = _CURRENT.get()
    if budget is None:
        return
    if (
        budget.crops + 1 > budget.max_crops
        or budget.crop_pixels + pixels > budget.max_pixels * 2
    ):
        exceeded(
            "Workflow crop count or cumulative crop pixels exceed the device budget"
        )
    budget.crops += 1
    budget.crop_pixels += pixels


def reserve_model_call():
    budget = _CURRENT.get()
    if budget is not None:
        if budget.model_calls >= 32:
            exceeded("Workflow exceeds 32 model invocations per image")
        budget.model_calls += 1


def reserve_image_output(pixels):
    budget = _CURRENT.get()
    if budget is None:
        return
    if not budget.allow_images:
        exceeded(
            "Camera Workflow results must contain JSON metadata; image encoding is disabled"
        )
    if budget.output_pixels + pixels > budget.max_pixels:
        exceeded("Workflow output images exceed the cumulative pixel budget")
    budget.output_pixels += pixels


def check_value(value):
    """Check logical expansion, counting repeated container references each time."""
    budget = _CURRENT.get()
    if budget is None:
        return
    stack, items, size = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        items += 1
        if items > 4096 or depth > 32:
            exceeded("Workflow intermediate JSON exceeds the item or nesting budget")
        if isinstance(item, (str, bytes)):
            size += len(item)
        elif isinstance(item, dict):
            if len(item) > 4096:
                exceeded("Workflow dictionary exceeds the item budget")
            stack.extend((v, depth + 1) for v in item.values())
            size += sum(len(str(key)) for key in item)
        elif isinstance(item, (list, tuple)):
            if len(item) > 4096:
                exceeded("Workflow sequence exceeds the item budget")
            stack.extend((v, depth + 1) for v in item)
        else:
            size += 16
        if size > budget.max_json_bytes:
            exceeded("Workflow intermediate JSON exceeds the byte budget")


def reserve_operation(value):
    budget = _CURRENT.get()
    if budget is not None:
        if budget.operations >= 4096:
            exceeded("Workflow exceeds the query operation budget")
        budget.operations += 1
        check_value(value)


def encode_bounded_json(value, max_bytes):
    def default(item):
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, Enum):
            return item.value
        # Original Workflow metadata can contain NumPy scalar values. Arrays
        # remain forbidden here: image conversion belongs to the image serializer.
        if (
            type(item).__module__.startswith("numpy")
            and getattr(item, "ndim", None) == 0
        ):
            return item.item()
        raise TypeError(
            f"Object of type {type(item).__name__} is not JSON serializable"
        )

    chunks, total = [], 0
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=default
    )
    for part in encoder.iterencode(value):
        raw = part.encode("utf-8")
        total += len(raw)
        if total > max_bytes:
            exceeded("JSON result exceeds the device byte budget")
        chunks.append(raw)
    return b"".join(chunks)
