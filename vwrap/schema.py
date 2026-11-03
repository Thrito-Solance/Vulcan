"""Typed V-Wrap request and response schema. Lengths are SI meters."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .errors import RequestValidationError


class FeatureKind(str, Enum):
    BOX = "box"
    THROUGH_BORE = "through_bore"
    SPHERE_EXCLUSION = "sphere_exclusion"
    ROUNDED_BOX = "rounded_box"
    CHAMFER = "chamfer"
    PATTERN = "pattern"
    FILLET_EDGE = "fillet_edge"
    THREADED_HOLE = "threaded_hole"
    RIB = "rib"
    FREEFORM = "freeform"


class ExportPath(str, Enum):
    ANALYTIC = "analytic"
    LEVEL_SET = "level_set"


class RequestState(str, Enum):
    EMPTY = "empty"
    VALID = "valid"
    UNSATISFIABLE = "unsatisfiable"


class PathEffect(str, Enum):
    INITIALIZES = "initializes"
    PRESERVES = "preserves"
    DOWNGRADES = "downgrades"
    BREAKS = "breaks"
    NO_OP = "no-op"


_AXES = frozenset({"x", "y", "z"})
_PARAMETER_KEYS = {
    FeatureKind.BOX: frozenset({"center", "half_extents"}),
    FeatureKind.THROUGH_BORE: frozenset({"axis", "center", "radius"}),
    FeatureKind.SPHERE_EXCLUSION: frozenset({"center", "radius"}),
    FeatureKind.ROUNDED_BOX: frozenset({"radius"}),
    FeatureKind.CHAMFER: frozenset({"normal", "offset"}),
    FeatureKind.PATTERN: frozenset({"axis", "radius", "offsets"}),
}


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise RequestValidationError(f"{name} must be a number; got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(
            f"{name} must be a number; got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise RequestValidationError(f"{name} must be finite; got {result!r}")
    return result


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise RequestValidationError(f"{name} must be positive; got {result!r}")
    return result


def _triple(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RequestValidationError(f"{name} must be a 3-vector; got {value!r}")
    return tuple(_finite(f"{name}[{index}]", item) for index, item in enumerate(value))  # type: ignore[return-value]


def _pair(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RequestValidationError(f"{name} must be a 2-vector; got {value!r}")
    return _finite(f"{name}[0]", value[0]), _finite(f"{name}[1]", value[1])


def _axis(name: str, value: Any) -> str:
    if not isinstance(value, str) or value not in _AXES:
        raise RequestValidationError(
            f"{name} must be one of {sorted(_AXES)}; got {value!r}"
        )
    return value


def _validate_params(kind: FeatureKind, params: dict[str, Any]) -> None:
    allowed = _PARAMETER_KEYS.get(kind)
    if allowed is None:
        if params:
            raise RequestValidationError(
                f"refused feature {kind.value!r} does not accept parameters"
            )
        return
    extra = set(params) - allowed
    if extra:
        raise RequestValidationError(
            f"unknown {kind.value} parameter(s): {sorted(extra)}"
        )

    if kind is FeatureKind.BOX:
        _triple("center", params.get("center"))
        half_extents = _triple("half_extents", params.get("half_extents"))
        if any(value <= 0.0 for value in half_extents):
            raise RequestValidationError("half_extents must all be positive")
    elif kind is FeatureKind.THROUGH_BORE:
        _axis("axis", params.get("axis"))
        _pair("center", params.get("center"))
        _positive("radius", params.get("radius"))
    elif kind is FeatureKind.SPHERE_EXCLUSION:
        _triple("center", params.get("center"))
        _positive("radius", params.get("radius"))
    elif kind is FeatureKind.ROUNDED_BOX:
        _positive("radius", params.get("radius"))
    elif kind is FeatureKind.CHAMFER:
        normal = _triple("normal", params.get("normal"))
        _finite("offset", params.get("offset"))
        if abs(math.hypot(*normal) - 1.0) > 1e-9:
            raise RequestValidationError("normal must be a unit vector")
    elif kind is FeatureKind.PATTERN:
        _axis("axis", params.get("axis"))
        _positive("radius", params.get("radius"))
        offsets = params.get("offsets")
        if not isinstance(offsets, (list, tuple)) or not offsets:
            raise RequestValidationError("pattern.offsets must be a non-empty list")
        for index, offset in enumerate(offsets):
            _pair(f"offsets[{index}]", offset)


@dataclass(frozen=True)
class Feature:
    kind: FeatureKind
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FeatureKind):
            raise RequestValidationError(f"unknown feature kind: {self.kind!r}")
        if not isinstance(self.params, dict):
            raise RequestValidationError("feature params must be an object")
        _validate_params(self.kind, self.params)


@dataclass(frozen=True)
class FeatureRequest:
    features: tuple[Feature, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple) or any(
            not isinstance(feature, Feature) for feature in self.features
        ):
            raise RequestValidationError("features must be a tuple of Feature values")

    def kinds(self) -> tuple[FeatureKind, ...]:
        return tuple(feature.kind for feature in self.features)


@dataclass(frozen=True)
class Fidelity:
    mode: Literal["exact", "bounded"]
    ratio: float | None = None
    voxel_size_m: float | None = None
    min_characteristic_dim_m: float | None = None


@dataclass(frozen=True)
class FeatureEffect:
    kind: FeatureKind
    voxels_changed: int
    landed: bool


@dataclass(frozen=True)
class TranslationReport:
    path: ExportPath
    accepted: bool
    primitives: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateAddition:
    kind: FeatureKind
    effect: PathEffect
    compatible_paths_after: frozenset[ExportPath]


@dataclass(frozen=True)
class ActionDescriptor:
    verb: Literal["add", "remove"]
    kind: FeatureKind
    effect: PathEffect | None
    rationale: str


@dataclass(frozen=True)
class DesignSpace:
    state: RequestState
    compatible_paths: frozenset[ExportPath]
    next_actions: tuple[ActionDescriptor, ...]
    candidate_additions: tuple[CandidateAddition, ...]
    blocking_reasons: tuple[str, ...]
    refused_categories: tuple[tuple[str, str], ...]
