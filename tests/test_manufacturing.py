from __future__ import annotations

import numpy as np
import pytest

from acwm import TriangleMesh, box_sdf, make_grid, mesh_from_level_set
from acwm_edge import MacroMicroGovernor
from acwm_manufacturing import (
    ManufacturingConfig,
    ManufacturingGovernor,
    ManufacturingInputError,
    surface_normals_from_sdf,
)


def test_filtered_sdf_normals_are_finite_and_normalized() -> None:
    spacing = (0.1, 0.1, 0.1)
    origin = (-0.7, -0.7, -0.7)
    phi = box_sdf(make_grid((15, 15, 15), spacing, origin), (0.4, 0.4, 0.4))
    mesh = mesh_from_level_set(phi, spacing, origin)

    normals = surface_normals_from_sdf(phi, spacing, origin, mesh)

    assert normals.shape == mesh.vertices.shape
    assert np.all(np.isfinite(normals))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)


def test_singular_sdf_gradient_fails_closed() -> None:
    phi = np.zeros((5, 5, 5))
    vertices = np.asarray([[2.0, 2.0, 2.0]])
    mesh = TriangleMesh(vertices, np.empty((0, 3), dtype=np.int64))

    with pytest.raises(ManufacturingInputError, match="gradient is singular"):
        surface_normals_from_sdf(
            phi, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), mesh
        )


def test_manufacturing_governor_rejects_directional_undercuts() -> None:
    governor = ManufacturingGovernor(
        ManufacturingConfig(check_cnc=True, check_additive=False)
    )

    accepted, reason = governor.evaluate(
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    )

    assert not accepted
    assert "undercut" in reason


def test_manufacturing_governor_enforces_additive_overhang_angle() -> None:
    governor = ManufacturingGovernor(
        ManufacturingConfig(check_cnc=False, check_additive=True)
    )

    accepted, reason = governor.evaluate(
        np.asarray([[np.sqrt(0.19), 0.0, -0.9], [0.0, 0.0, 1.0]])
    )

    assert not accepted
    assert "overhang" in reason


def test_macro_governor_requires_normals_when_manufacturing_is_enabled() -> None:
    manufacturing = ManufacturingGovernor(
        ManufacturingConfig(check_cnc=False, check_additive=True)
    )
    governor = MacroMicroGovernor(100.0, manufacturing)

    accepted, reason = governor.evaluate_and_filter(np.asarray([10.0]), True)

    assert not accepted
    assert "normals are missing" in reason
