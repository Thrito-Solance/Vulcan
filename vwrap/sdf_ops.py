"""SDF primitives and combinators for the bounded level-set path."""
from __future__ import annotations

import numpy as np


def _grid(points):
    return points[..., 0], points[..., 1], points[..., 2]


def cylinder_sdf(points, axis: str, center_uv, radius: float):
    x, y, z = _grid(points)
    u, v = center_uv
    if axis == "z":
        du, dv = x - u, y - v
    elif axis == "y":
        du, dv = x - u, z - v
    elif axis == "x":
        du, dv = y - u, z - v
    else:
        raise ValueError(f"axis must be x, y, or z; got {axis!r}")
    return np.sqrt(du * du + dv * dv) - radius


def half_space_sdf(points, normal, offset: float):
    nx, ny, nz = normal
    x, y, z = _grid(points)
    return nx * x + ny * y + nz * z - offset


def rounded_box_sdf(points, half_extents, radius: float):
    half = np.asarray(half_extents, dtype=float)
    if half.shape != (3,) or np.any(half <= 0.0):
        raise ValueError("half_extents must be three positive values")
    if not np.isfinite(radius) or radius <= 0.0 or radius > float(np.min(half)):
        raise ValueError("radius must be positive and no greater than the smallest half extent")
    x, y, z = _grid(points)
    core = half - radius
    qx = np.abs(x) - core[0]
    qy = np.abs(y) - core[1]
    qz = np.abs(z) - core[2]
    outer = np.sqrt(
        np.maximum(qx, 0.0) ** 2
        + np.maximum(qy, 0.0) ** 2
        + np.maximum(qz, 0.0) ** 2
    )
    inner = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
    return outer + inner - radius


def union(a, b):
    return np.minimum(a, b)


def subtract(a, b):
    return np.maximum(a, -b)
