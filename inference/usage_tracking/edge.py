"""Local RKNN execution has no cloud usage queue or telemetry worker threads."""

import inspect
from functools import wraps
from typing import Callable, TypeVar

Function = TypeVar("Function", bound=Callable)
_USAGE_ARGUMENTS = frozenset(
    {
        "usage_fps",
        "usage_api_key",
        "usage_workflow_id",
        "usage_workflow_preview",
        "usage_inference_test_run",
        "usage_billable",
    }
)


def usage_collector(category: str) -> Callable[[Function], Function]:
    """Preserve decorator call semantics without retaining usage payloads.

    The original decorator consumes these keyword-only control arguments before
    invoking the decorated function. Simply returning that function would leak
    the controls into workflow execution and raise an unexpected-keyword error.
    """

    def decorate(function: Function) -> Function:
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args, **kwargs):
                for key in _USAGE_ARGUMENTS:
                    kwargs.pop(key, None)
                return await function(*args, **kwargs)

            return async_wrapper

        @wraps(function)
        def wrapper(*args, **kwargs):
            for key in _USAGE_ARGUMENTS:
                kwargs.pop(key, None)
            return function(*args, **kwargs)

        return wrapper

    return decorate
