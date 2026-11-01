from __future__ import annotations

import numpy as np
import pytest

from acwm import box_sdf, make_grid
from acwm_edge import MacroContext
from acwm_sensitivity import (
    SensitivityInputError,
    SensitivityInvariantError,
    run_sensitivity_sweep,
)


def _baseline() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spacing = np.full(3, 0.1)
    origin = np.full(3, -0.5)
    points = make_grid((11, 11, 11), spacing, origin)
    phi = box_sdf(points, (0.4, 0.4, 0.4))
    velocity = np.full(phi.shape, -1.0)
    return phi, velocity, spacing, origin


def _sweep(phi, velocity, spacing, origin, **overrides):
    arguments = {
        "dt": 0.02,
        "load_vectors_n": ((0.0, 0.0, 10.0),),
        "support_faces": ("bottom",),
        "load_face": "top",
        "young_modulus_pa": 200e9,
        "poisson_ratio": 0.3,
        "allowable_stresses_pa": (1e9,),
        "helmholtz_radii_m": (0.0,),
        "ks_rhos": (100.0,),
        "macro_contexts": (MacroContext(0.0, 1.0, 1.0),),
    }
    arguments.update(overrides)
    return run_sensitivity_sweep(
        phi, velocity, spacing, origin, **arguments
    )


def test_sweep_runs_fem_per_candidate_and_crosses_all_sensitivity_axes():
    phi, velocity, spacing, origin = _baseline()
    original = phi.copy()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        load_vectors_n=((0.0, 0.0, 10.0), (0.0, 0.0, 20.0)),
        allowable_stresses_pa=(1e9,),
        helmholtz_radii_m=(0.0, 0.05),
        ks_rhos=(50.0, 500.0),
        macro_contexts=(
            MacroContext(0.0, 0.5, 0.5),
            MacroContext(0.1, 1.5, 1.0),
        ),
    )

    assert len(report.cases) == 16
    assert report.baseline_unchanged
    np.testing.assert_array_equal(phi, original)
    assert all(case.status == "accepted" for case in report.cases)
    assert all(case.surface_validation is not None for case in report.cases)
    assert all(case.surface_validation.valid for case in report.cases)
    assert all(case.maximum_von_mises_pa is not None for case in report.cases)
    assert all(case.ks_von_mises_pa >= case.maximum_von_mises_pa for case in report.cases)
    for case in report.cases:
        assert case.candidate_sha256 is not None
        assert case.committed_sha256 == case.candidate_sha256
        assert case.removed_voxels > 0


def test_low_allowable_rolls_back_without_mutating_the_baseline():
    phi, velocity, spacing, origin = _baseline()
    original = phi.copy()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        allowable_stresses_pa=(1e-12,),
    )

    assert len(report.cases) == 1
    case = report.cases[0]
    assert not case.accepted
    assert case.status == "rollback"
    assert "exceeds" in case.reason
    assert case.candidate_sha256 is not None
    assert case.committed_sha256 == report.baseline_sha256
    assert case.removed_voxels is not None
    np.testing.assert_array_equal(phi, original)


def test_large_load_is_rejected_by_governor_and_ks():
    phi, velocity, spacing, origin = _baseline()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        load_vectors_n=((0.0, 0.0, 1e12),),
        allowable_stresses_pa=(1.0,),
    )
    case = report.cases[0]
    assert not case.accepted
    assert case.status == "rollback"
    assert "exceeds allowable" in case.reason
    assert case.committed_sha256 == report.baseline_sha256


def test_macro_context_stress_scale_changes_governed_stress():
    phi, velocity, spacing, origin = _baseline()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        macro_contexts=(
            MacroContext(0.0, 0.5, 1.0),
            MacroContext(0.0, 2.0, 1.0),
        ),
    )
    low, high = report.cases
    assert low.maximum_von_mises_pa is not None
    assert high.maximum_von_mises_pa is not None
    assert high.maximum_von_mises_pa == pytest.approx(
        4.0 * low.maximum_von_mises_pa
    )


def test_load_scale_helmholtz_radius_and_ks_rho_change_outputs():
    phi, _, spacing, origin = _baseline()
    coordinates = make_grid(phi.shape, spacing, origin)
    velocity = np.where(phi <= 0, -(1.0 + coordinates[..., 0]), 0.0)
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        load_vectors_n=((0.0, 0.0, 10.0), (0.0, 0.0, 20.0)),
        helmholtz_radii_m=(0.0, 0.05),
        ks_rhos=(50.0, 500.0),
    )
    cases = {
        (
            case.load_vector_n[2],
            case.helmholtz_radius_m,
            case.ks_rho,
        ): case
        for case in report.cases
    }
    low_load = cases[(10.0, 0.0, 50.0)]
    high_load = cases[(20.0, 0.0, 50.0)]
    assert high_load.maximum_von_mises_pa == pytest.approx(
        2.0 * low_load.maximum_von_mises_pa, rel=1e-8
    )

    low_rho = cases[(10.0, 0.0, 50.0)]
    high_rho = cases[(10.0, 0.0, 500.0)]
    assert low_rho.ks_von_mises_pa > high_rho.ks_von_mises_pa

    unfiltered = cases[(10.0, 0.0, 50.0)]
    filtered = cases[(10.0, 0.05, 50.0)]
    assert unfiltered.filtered_velocity_l2 != pytest.approx(
        filtered.filtered_velocity_l2
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"load_vectors_n": ()},
        {"allowable_stresses_pa": (0.0,)},
        {"helmholtz_radii_m": (-0.1,)},
        {"ks_rhos": (0.0,)},
        {"macro_contexts": (MacroContext(-1.0, 1.0),)},
        {"macro_contexts": (MacroContext(0.0, 0.0),)},
        {"support_faces": ("top",)},
        {"normal_velocity": np.ones((11, 11, 11))},
        {"max_cases": 0},
    ),
)
def test_invalid_sweep_inputs_fail_before_evaluation(overrides):
    phi, velocity, spacing, origin = _baseline()
    sweep_overrides = dict(overrides)
    velocity = sweep_overrides.pop("normal_velocity", velocity)
    with pytest.raises(SensitivityInputError):
        _sweep(phi, velocity, spacing, origin, **sweep_overrides)


def test_invalid_baseline_fails_before_any_case_is_reported():
    phi, velocity, spacing, origin = _baseline()
    invalid = np.ones_like(phi)
    with pytest.raises(SensitivityInvariantError, match="baseline geometry is invalid"):
        _sweep(invalid, velocity, spacing, origin)


def test_sweep_case_limit_rejects_excessive_cartesian_products():
    phi, velocity, spacing, origin = _baseline()
    with pytest.raises(SensitivityInputError, match="exceeding max_cases=3"):
        _sweep(
            phi,
            velocity,
            spacing,
            origin,
            load_vectors_n=((0.0, 0.0, 10.0), (0.0, 0.0, 20.0)),
            allowable_stresses_pa=(1e8, 1e9),
            max_cases=3,
        )


def test_filter_and_context_breakdowns_are_recorded_as_rollbacks():
    phi, velocity, spacing, origin = _baseline()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        helmholtz_radii_m=(1e308,),
        macro_contexts=(MacroContext(0.0, 1.0, 1e6),),
    )
    case = report.cases[0]
    assert not case.accepted
    assert case.status == "rollback"
    assert "Helmholtz radius squared" in case.reason
    assert case.committed_sha256 == report.baseline_sha256


def test_macro_context_cannot_overshoot_the_stable_cell_time_step():
    phi, velocity, spacing, origin = _baseline()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        macro_contexts=(MacroContext(0.0, 1.0, 100.0),),
    )
    case = report.cases[0]
    assert not case.accepted
    assert "larger than half the smallest cell" in case.reason
    assert case.committed_sha256 == report.baseline_sha256


def test_extreme_finite_ks_rho_remains_a_valid_sensitivity_case():
    phi, velocity, spacing, origin = _baseline()
    report = _sweep(
        phi,
        velocity,
        spacing,
        origin,
        ks_rhos=(1e308,),
    )
    case = report.cases[0]
    assert case.accepted
    assert np.isfinite(case.ks_von_mises_pa)
