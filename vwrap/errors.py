"""Structured refusal taxonomy for V-Wrap."""


class VWrapError(Exception):
    """Base class for V-Wrap errors."""


class RequestValidationError(VWrapError):
    """A request does not conform to the V-Wrap schema."""


class UnsupportedFeatureError(VWrapError):
    """A feature is outside the explicitly supported capability boundary."""

    def __init__(
        self,
        feature_kind: str,
        reason: str,
        supported_alternative: str | None = None,
    ) -> None:
        self.feature_kind = feature_kind
        self.reason = reason
        self.supported_alternative = supported_alternative
        message = f"unsupported feature '{feature_kind}': {reason}"
        if supported_alternative:
            message += f" (try '{supported_alternative}' instead)"
        super().__init__(message)


class TranslationError(VWrapError):
    """The deterministic translation airlock refused to emit a primitive."""


class ExportError(VWrapError):
    """Export failed after translation succeeded."""
