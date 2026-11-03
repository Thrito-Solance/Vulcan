"""Deterministic translation airlock between requests and Vulcan primitives."""
from __future__ import annotations

from acwm_cad import BoxWithThroughBores, ThroughBore

from .capability import ANALYTIC_SUPPORTED, LEVEL_SET_SUPPORTED
from .errors import TranslationError
from .features import (
    AnalyticPrimitives,
    LevelSetField,
    expand_analytic,
    expand_level_set,
)
from .schema import ExportPath, FeatureRequest, TranslationReport


def _to_box_with_bores(primitives: AnalyticPrimitives) -> BoxWithThroughBores:
    if primitives.bore is None:
        return BoxWithThroughBores(
            center=primitives.center,
            half_extents=primitives.half_extents,
        )
    axis, center_uv, radius = primitives.bore
    return BoxWithThroughBores(
        center=primitives.center,
        half_extents=primitives.half_extents,
        bores=(ThroughBore(axis, center_uv, radius),),
    )


def translate(
    request: FeatureRequest,
    *,
    path: ExportPath,
    resolution: int = 32,
):
    if not request.features:
        raise TranslationError("feature request is empty")
    if not isinstance(path, ExportPath):
        raise TranslationError(f"unknown export path: {path!r}")

    if path is ExportPath.ANALYTIC:
        if not all(feature.kind in ANALYTIC_SUPPORTED for feature in request.features):
            raise TranslationError(
                "analytic path requires only a box and at most one through-bore; "
                f"got {tuple(kind.value for kind in request.kinds())}"
            )
        primitives = expand_analytic(request)
        report = TranslationReport(
            path=ExportPath.ANALYTIC,
            accepted=True,
            primitives=("box",) + (("through_bore",) if primitives.bore else ()),
        )
        return "analytic", _to_box_with_bores(primitives), report

    if path is ExportPath.LEVEL_SET:
        if not all(feature.kind in LEVEL_SET_SUPPORTED for feature in request.features):
            raise TranslationError(
                "level-set path cannot represent this request; "
                f"kinds={tuple(kind.value for kind in request.kinds())}"
            )
        field: LevelSetField = expand_level_set(request, resolution=resolution)
        report = TranslationReport(
            path=ExportPath.LEVEL_SET,
            accepted=True,
            primitives=tuple(feature.kind.value for feature in request.features),
            warnings=(
                (
                    "level-set output is a faceted tessellation, not an analytic "
                    "primitive, and is not verified by the FEA loop"
                ),
            ),
        )
        return "level_set", field, report

    raise TranslationError(f"unknown export path: {path!r}")
