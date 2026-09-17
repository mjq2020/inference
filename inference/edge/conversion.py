"""Injection boundary for a future conversion service; no assumed wire protocol.

Conversion runs outside this application. An implementation must obtain a model
package for the requested platform. Installing that package and granting its
path/digest access in appmgr are separate deployment operations.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from inference.edge.errors import EdgeError


@dataclass(frozen=True)
class ConversionRequest:
    source_reference: str
    target_platform: str = "rv1126b"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversionResult:
    state: str
    job_reference: str | None = None
    package_reference: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


class Converter(Protocol):
    def convert(self, request: ConversionRequest) -> ConversionResult: ...


class ConversionService:
    def __init__(self, converter: Converter | None = None):
        self._converter = converter

    @property
    def configured(self) -> bool:
        return self._converter is not None

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "state": "configured" if self.configured else "unconfigured",
            "target_platform": "rv1126b",
            "automatic_installation": False,
        }

    def convert(self, request: ConversionRequest) -> ConversionResult:
        if self._converter is None:
            raise EdgeError(
                "No model conversion provider is configured.",
                code="converter_unconfigured",
                status_code=503,
            )
        if request.target_platform != "rv1126b" or not request.source_reference:
            raise EdgeError(
                "Conversion requires a source reference and target rv1126b.",
                code="invalid_conversion_request",
            )
        return self._converter.convert(request)
