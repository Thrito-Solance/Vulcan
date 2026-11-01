from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import MatrixRankWarning

from acwm import (
    HexMesh,
    TriangleMesh,
    box_sdf,
    carve_exclusions,
    export_mesh,
    helmholtz_filter,
    ks_stress,
    level_set_step,
    make_grid,
    mesh_from_level_set,
    reinitialize_sdf,
    solve_linear_elasticity,
    sphere_sdf,
    tolerance_field,
    validate_mesh,
)


def test_sdf_boolean_and_mesh_topology():
    spacing = np.full(3, 0.1)
    origin = np.full(3, -0.5)
    points = make_grid((11, 11, 11), spacing, origin)
    domain = box_sdf(points, (0.4, 0.4, 0.4))
    exclusion = sphere_sdf(points, (0.0, 0.0, 0.0), 0.2)
    carved = carve_exclusions(domain, [exclusion])
    assert carved[5, 5, 5] > 0
    assert carved[0, 0, 0] > 0

    mesh = mesh_from_level_set(carved, spacing, origin)
    result = validate_mesh(mesh)
    assert result.valid, result.issues
    assert result.watertight
    assert result.winding_consistent
    assert result.vertex_manifold
    assert result.components == 2
    assert result.euler_characteristic == 4
    assert result.genus == 0


def test_validator_rejects_open_and_nonmanifold_meshes():
    tetra_vertices = np.asarray(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float
    )
    open_tetra = TriangleMesh(tetra_vertices, np.asarray([(0, 2, 1), (0, 1, 3), (0, 3, 2)]))
    assert not validate_mesh(open_tetra).valid
    nonmanifold = TriangleMesh(
        tetra_vertices,
        np.asarray([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3), (0, 1, 2)]),
    )
    assert not validate_mesh(nonmanifold).valid


def test_helmholtz_filter_smooths_and_preserves_constant_field():
    constant = np.ones((5, 5, 5))
    np.testing.assert_allclose(helmholtz_filter(constant, 0.5), constant, atol=1e-8)
    impulse = np.zeros_like(constant)
    impulse[2, 2, 2] = 1.0
    filtered = helmholtz_filter(impulse, 0.5)
    assert filtered[2, 2, 2] < 1.0
    assert np.count_nonzero(filtered > 0) > 1


def test_level_set_step_and_reinitialization():
    points = make_grid((9, 9, 9), (0.1, 0.1, 0.1), (-0.4, -0.4, -0.4))
    phi = sphere_sdf(points, (0, 0, 0), 0.2)
    expanded = level_set_step(phi, 0.1, 0.1, (0.1, 0.1, 0.1))
    assert expanded[6, 4, 4] < phi[6, 4, 4]
    reinitialized = reinitialize_sdf(expanded, (0.1, 0.1, 0.1))
    assert np.all(np.isfinite(reinitialized))
    assert np.array_equal(reinitialized <= 0, expanded <= 0)


def test_ks_is_stable_and_upper_bounds_discrete_peak():
    stresses = np.asarray([1.0, 2.0, 3.0])
    aggregate = ks_stress(stresses, rho=1000)
    assert aggregate >= np.max(stresses)
    assert aggregate == pytest.approx(3.0, abs=0.002)
    assert ks_stress(np.full(4, 1e6), rho=1000) == pytest.approx(
        1e6 + np.log(4) / 1000
    )
    assert np.isfinite(ks_stress([1e307, 1e307], rho=1000))


def test_ks_handles_extreme_finite_rho_without_overflowing_intermediates():
    assert ks_stress([0.0, 3.0], rho=1e308) == pytest.approx(3.0)
    with pytest.raises(OverflowError, match="finite float range"):
        ks_stress([1.0, 2.0], rho=1e-320)


def test_tolerance_guards_zero_sensitivity_and_expansion_sign():
    tolerance = tolerance_field(
        np.asarray([0.0, 4.0]),
        np.asarray([10.0, -10.0]),
        np.asarray([2.0, 2.0]),
        alpha_cte=1e-5,
        beta_1=1.0,
        beta_2=1.0,
    )
    assert np.all(np.isfinite(tolerance))
    assert np.all(tolerance > 0)
    with pytest.raises(ValueError):
        tolerance_field([1.0], [1.0], [-1.0], 1e-5, 1.0, 1.0)


def test_voxel_fem_solves_a_clamped_single_hex():
    mesh = HexMesh(
        nodes=np.asarray(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ],
            dtype=float,
        ),
        elements=np.arange(8, dtype=np.int64).reshape(1, 8),
    )
    fixed = np.asarray([0, 3, 4, 7])
    force = np.zeros((8, 3))
    force[[1, 2, 5, 6], 0] = 0.25
    result = solve_linear_elasticity(
        mesh,
        young_modulus=200e9,
        poisson_ratio=0.3,
        fixed_nodes=fixed,
        forces=force,
    )
    assert np.all(np.isfinite(result.displacement))
    assert np.all(result.displacement[fixed] == 0)
    assert np.all(result.displacement[[1, 2, 5, 6], 0] > 0)
    assert result.maximum_von_mises > 0


def test_fem_rejects_noncanonical_meshes_and_noninteger_indices():
    nodes = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype=float,
    )
    elements = np.arange(8, dtype=np.int64).reshape(1, 8)
    forces = np.zeros((8, 3))
    with pytest.raises(ValueError, match="distinct nodes"):
        solve_linear_elasticity(
            HexMesh(nodes, np.asarray([[0, 0, 2, 3, 4, 5, 6, 7]])),
            200e9,
            0.3,
            [0, 3, 4, 7],
            forces,
        )
    with pytest.raises(ValueError, match="axis-aligned"):
        solve_linear_elasticity(
            HexMesh(nodes + np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0, 0], [0, 0, 0],
                                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]), elements),
            200e9,
            0.3,
            [0, 3, 4, 7],
            forces,
        )
    with pytest.raises(ValueError, match="indices must be integers"):
        solve_linear_elasticity(
            HexMesh(nodes, elements.astype(float)),
            200e9,
            0.3,
            [0, 3, 4, 7],
            forces,
        )
    with pytest.raises(ValueError, match="indices must be integers"):
        solve_linear_elasticity(
            HexMesh(nodes, elements),
            200e9,
            0.3,
            [0.5, 3, 4, 7],
            forces,
        )


def test_fem_promotes_singular_warning_to_explicit_failure(monkeypatch):
    import acwm

    nodes = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype=float,
    )
    mesh = HexMesh(nodes, np.arange(8, dtype=np.int64).reshape(1, 8))

    def singular_solve(*args, **kwargs):
        raise MatrixRankWarning("singular matrix")

    monkeypatch.setattr(acwm, "spsolve", singular_solve)
    with pytest.raises(ValueError, match="unstable model") as error:
        solve_linear_elasticity(
            mesh,
            200e9,
            0.3,
            [0, 3, 4, 7],
            np.ones((8, 3)),
        )
    assert isinstance(error.value.__cause__, MatrixRankWarning)


def test_mesh_export_fails_closed_for_invalid_mesh_and_cad(tmp_path):
    points = make_grid((9, 9, 9), (0.1, 0.1, 0.1), (-0.4, -0.4, -0.4))
    mesh = mesh_from_level_set(box_sdf(points, (0.25, 0.25, 0.25)), (0.1,) * 3, (-0.4,) * 3)
    output = export_mesh(mesh, tmp_path / "cube.obj")
    assert output.exists()
    with pytest.raises(NotImplementedError, match="B-rep"):
        export_mesh(mesh, tmp_path / "cube.step")
    with pytest.raises(ValueError, match="invalid mesh"):
        export_mesh(TriangleMesh(mesh.vertices, mesh.faces[:-1]), tmp_path / "open.stl")
