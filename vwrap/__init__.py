"""V-Wrap: explicit application-layer translation boundary for Vulcan."""

from .errors import (
    ExportError,
    RequestValidationError,
    TranslationError,
    UnsupportedFeatureError,
    VWrapError,
)
from .exporter import ExportResult, export
from .feasibility import query_design_space
from .schema import (
    ActionDescriptor,
    CandidateAddition,
    DesignSpace,
    ExportPath,
    Feature,
    FeatureEffect,
    FeatureKind,
    FeatureRequest,
    Fidelity,
    PathEffect,
    RequestState,
    TranslationReport,
)
from .translator import translate

__all__ = [
    "ActionDescriptor",
    "CandidateAddition",
    "DesignSpace",
    "ExportError",
    "ExportPath",
    "ExportResult",
    "Feature",
    "FeatureEffect",
    "FeatureKind",
    "FeatureRequest",
    "Fidelity",
    "PathEffect",
    "RequestState",
    "RequestValidationError",
    "TranslationError",
    "TranslationReport",
    "UnsupportedFeatureError",
    "VWrapError",
    "export",
    "query_design_space",
    "translate",
]
