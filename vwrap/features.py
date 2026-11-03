"""Expand validated feature requests into analytic primitives or sampled SDFs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acwm import box_sdf, make_grid, sphere_sdf

from .capability import (
    ANALYTIC_SUPPORTED,
    FIDELITY_FLOOR,
    LEVEL_SET_SUPPORTED,
    REFUSED_WITH_REASON,
)
from .errors import UnsupportedFeatureError
from .schema import Feature, FeatureEffect, FeatureKind, FeatureRequest, Fidelity
from .sdf_ops import cylinder_sdf, half_space_sdf, rounded_box_sdf, subtract


@dataclass(frozen=True)
class AnalyticPrimitives:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    bore: tuple[str, tuple[float, float], float] | None


@dataclass(frozen=True)
class LevelSetField:
    phi: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    fidelity: Fidelity
    landed: tuple[FeatureEffect, ...]


def _refuse(
    kind: FeatureKind,
    reason: str,
    alternative: str | None = None,
) -> None:
    raise UnsupportedFeatureError(kind.value, reason, alternative)


def _count(features: tuple[Feature, ...], kind: FeatureKind) -> int:
    return sum(feature.kind is kind for feature in features)


def expand_analytic(request: FeatureRequest) -> AnalyticPrimitives:
    boxes = [feature for feature in request.features if feature.kind is FeatureKind.BOX]
    bores = [
        feature
        for feature in request.features
        if feature.kind is FeatureKind.THROUGH_BORE
    ]

    for feature in request.features:
        if feature.kind not in ANALYTIC_SUPPORTED:
            _refuse(
                feature.kind,
                REFUSED_WITH_REASON.get(
                    feature.kind, "not in Vulcan's analytic family"
                ),
            )
    if len(boxes) != 1:
        _refuse(
            FeatureKind.BOX,
            f"analytic path requires exactly one box; got {len(boxes)}",
        )
    if len(bores) > 1:
        _refuse(
            FeatureKind.THROUGH_BORE,
            f"analytic path allows at most one bore; got {len(bores)}",
        )

    box = boxes[0]
    bore = bores[0] if bores else None
    return AnalyticPrimitives(
        center=tuple(box.params["center"]),
        half_extents=tuple(box.params["half_extents"]),
        bore=(
            (
                bore.params["axis"],
                tuple(bore.params["center"]),
                float(bore.params["radius"]),
            )
            if bore
            else None
        ),
    )


def _cross_section_voxel_size(spacing, axis: str | None) -> float:
    if axis is None:
        return max(spacing)
    if axis == "z":
        return max(spacing[0], spacing[1])
    if axis == "y":
        return max(spacing[0], spacing[2])
    if axis == "x":
        return max(spacing[1], spacing[2])
    raise ValueError(f"unknown axis: {axis!r}")


def _characteristic_dim(feature: Feature) -> tuple[float, str | None]:
    kind, params = feature.kind, feature.params
    if kind in (FeatureKind.THROUGH_BORE, FeatureKind.PATTERN):
        return 2.0 * float(params["radius"]), params["axis"]
    if kind is FeatureKind.SPHERE_EXCLUSION:
        return 2.0 * float(params["radius"]), None
    if kind in (FeatureKind.ROUNDED_BOX, FeatureKind.CHAMFER):
        return abs(float(params["radius" if kind is FeatureKind.ROUNDED_BOX else "offset"])), None
    raise AssertionError(f"no characteristic dimension for {kind}")


def _resolve_feature(
    grid: np.ndarray,
    phi: np.ndarray,
    feature: Feature,
) -> tuple[np.ndarray, FeatureEffect]:
    kind, params = feature.kind, feature.params
    if kind is FeatureKind.THROUGH_BORE:
        after = subtract(
            phi,
            cylinder_sdf(
                grid,
                params["axis"],
                tuple(params["center"]),
                float(params["radius"]),
            ),
        )
    elif kind is FeatureKind.SPHERE_EXCLUSION:
        after = subtract(
            phi,
            sphere_sdf(
                grid,
                tuple(params["center"]),
                float(params["radius"]),
            ),
        )
    elif kind is FeatureKind.CHAMFER:
        after = np.maximum(
            phi,
            half_space_sdf(
                grid, tuple(params["normal"]), float(params["offset"])
            ),
        )
    elif kind is FeatureKind.PATTERN:
        after = phi
        for offset in params["offsets"]:
            after = subtract(
                after,
                cylinder_sdf(
                    grid,
                    params["axis"],
                    tuple(offset),
                    float(params["radius"]),
                ),
            )
    else:
        _refuse(kind, "internal capability mismatch")
    changed = int(np.count_nonzero((phi <= 0.0) != (after <= 0.0)))
    return after, FeatureEffect(kind, changed, changed > 0)


def expand_level_set(
    request: FeatureRequest,
    *,
    resolution: int,
) -> LevelSetField:
    if (
        not isinstance(resolution, int)
        or isinstance(resolution, bool)
        or resolution < 4
    ):
        _refuse(FeatureKind.BOX, f"resolution must be an integer >= 4; got {resolution!r}")
    for feature in request.features:
        if feature.kind not in LEVEL_SET_SUPPORTED:
            _refuse(
                feature.kind,
                REFUSED_WITH_REASON.get(
                    feature.kind, "not supported on the level-set path"
                ),
            )

    boxes = [feature for feature in request.features if feature.kind is FeatureKind.BOX]
    if len(boxes) != 1:
        _refuse(
            FeatureKind.BOX,
            f"level-set path requires exactly one base box; got {len(boxes)}",
        )
    base = boxes[0]
    center = np.asarray(base.params["center"], dtype=float)
    half = np.asarray(base.params["half_extents"], dtype=float)
    rounded = [
        feature
        for feature in request.features
        if feature.kind is FeatureKind.ROUNDED_BOX
    ]
    if len(rounded) > 1:
        _refuse(FeatureKind.ROUNDED_BOX, "at most one rounded_box may replace the base box")
    if rounded and float(rounded[0].params["radius"]) > float(np.min(half)):
        _refuse(
            FeatureKind.ROUNDED_BOX,
            "corner radius must not exceed the smallest base-box half extent",
        )

    pad = 3.0 * float(np.max(half)) / (resolution - 1)
    spacing = tuple(float((half[index] + pad) * 2.0 / (resolution - 1)) for index in range(3))
    origin = tuple(float(center[index] - half[index] - pad) for index in range(3))
    for feature in request.features:
        if feature.kind in (FeatureKind.BOX, FeatureKind.ROUNDED_BOX):
            continue
        dimension, axis = _characteristic_dim(feature)
        voxel_size = _cross_section_voxel_size(spacing, axis)
        if dimension < FIDELITY_FLOOR * voxel_size:
            needed_resolution = resolution
            while needed_resolution < 4096:
                needed_resolution += 1
                pad_n = 3.0 * float(np.max(half)) / (needed_resolution - 1)
                spacing_n = tuple(
                    float((half[index] + pad_n) * 2.0 / (needed_resolution - 1))
                    for index in range(3)
                )
                if dimension >= FIDELITY_FLOOR * _cross_section_voxel_size(
                    spacing_n, axis
                ):
                    break
            _refuse(
                feature.kind,
                f"characteristic dimension {dimension:.4g} m is below "
                f"{FIDELITY_FLOOR}x voxel size {voxel_size:.4g} m in its "
                "cross-section; feature will alias",
                alternative=f"resolution >= {needed_resolution}",
            )

    grid = make_grid((resolution, resolution, resolution), spacing, origin)
    local_grid = grid - center
    if rounded:
        phi = rounded_box_sdf(
            local_grid, tuple(half), float(rounded[0].params["radius"])
        )
        base_before = box_sdf(local_grid, tuple(half))
        changed = int(np.count_nonzero((base_before <= 0.0) != (phi <= 0.0)))
        landed = [FeatureEffect(FeatureKind.ROUNDED_BOX, changed, changed > 0)]
    else:
        phi = box_sdf(local_grid, tuple(half))
        landed = []

    for feature in request.features:
        if feature.kind in (FeatureKind.BOX, FeatureKind.ROUNDED_BOX):
            continue
        phi, effect = _resolve_feature(grid, phi, feature)
        landed.append(effect)

    for effect in landed:
        if not effect.landed:
            _refuse(
                effect.kind,
                "feature produced no measurable change in the field; it is "
                "outside the grid domain or below the resolution bound",
            )

    characteristic_dims = [
        _characteristic_dim(feature)[0]
        for feature in request.features
        if feature.kind not in (FeatureKind.BOX, FeatureKind.ROUNDED_BOX)
    ]
    ratios = [
        _characteristic_dim(feature)[0]
        / _cross_section_voxel_size(
            spacing, _characteristic_dim(feature)[1]
        )
        for feature in request.features
        if feature.kind not in (FeatureKind.BOX, FeatureKind.ROUNDED_BOX)
    ]
    if rounded:
        characteristic_dims.append(float(rounded[0].params["radius"]))
        ratios.append(
            float(rounded[0].params["radius"])
            / _cross_section_voxel_size(spacing, None)
        )
    fidelity = Fidelity(
        mode="bounded",
        ratio=min(ratios) if ratios else None,
        voxel_size_m=max(spacing),
        min_characteristic_dim_m=min(characteristic_dims) if characteristic_dims else None,
    )
    return LevelSetField(
        phi=phi,
        spacing=spacing,
        origin=origin,
        fidelity=fidelity,
        landed=tuple(landed),
    )
