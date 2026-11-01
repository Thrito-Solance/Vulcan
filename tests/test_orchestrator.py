from __future__ import annotations

import json

import pytest

from acwm_cad import BoxWithThroughBores
from acwm_edge import TinyEdgeSurrogate
from acwm_manufacturing import ManufacturingConfig
from acwm_orchestrator import (
    EdgeInferenceError,
    GenerationConfig,
    IntentConstraintError,
    MaterialProperties,
    PhysicsConstraintError,
    generate,
    main,
)

_INTENT = (
    "Steel bracket with 60 x 60 x 60 mm envelope, downward 10 N load "
    "applied to top face, fixed at bottom face."
)
_MATERIAL = MaterialProperties(
    grade="explicit test steel",
    young_modulus_pa=200e9,
    poisson_ratio=0.3,
    allowable_stress_pa=1e9,
)


def test_closed_loop_generates_verified_cad_and_reports_rollback_or_acceptance(
    tmp_path,
):
    pytest.importorskip("cadquery")
    from cadquery import importers

    output = tmp_path / "bracket.step"
    result = generate(
        _INTENT,
        _MATERIAL,
        output,
        config=GenerationConfig(resolution=10, max_iterations=2),
    )

    assert output.is_file()
    shape = importers.importStep(str(output)).val()
    assert shape.isValid()
    assert len(shape.Solids()) == 1
    assert result["final_volume_m3"] == pytest.approx(shape.Volume(), rel=1e-9)
    assert result["status"] == "verified exploratory export"
    assert result["surface_validation"]["valid"]
    assert result["manufacturing"]["enabled"] is False
    assert result["ks_von_mises_pa"] <= _MATERIAL.allowable_stress_pa
    assert result["surrogate_preview"] == {
        "enabled": False,
        "max_abs_raw_outputs": [],
        "authoritative": False,
    }
    assert len(result["history"]) <= 2 * 4
    for event in result["history"]:
        assert event["status"] in {"accepted", "rollback"}
        if not event["accepted"]:
            assert event["reason"]


def test_failed_initial_fea_does_not_write_a_cad_file(tmp_path):
    output = tmp_path / "overstressed.step"
    material = MaterialProperties(
        grade="explicit test steel",
        young_modulus_pa=200e9,
        poisson_ratio=0.3,
        allowable_stress_pa=1e-12,
    )
    with pytest.raises(PhysicsConstraintError, match="initial design exceeds"):
        generate(
            _INTENT,
            material,
            output,
            config=GenerationConfig(resolution=10, max_iterations=1),
        )
    assert not output.exists()


def test_manufacturing_governor_vetoes_initial_design_before_export(tmp_path):
    output = tmp_path / "unmachinable.step"

    with pytest.raises(PhysicsConstraintError, match="Manufacturing Governor"):
        generate(
            _INTENT,
            _MATERIAL,
            output,
            config=GenerationConfig(
                resolution=10,
                max_iterations=0,
                manufacturing=ManufacturingConfig(
                    check_cnc=True, check_additive=False
                ),
            ),
        )

    assert not output.exists()


def test_untrained_edge_surrogate_falls_back_to_global_loop(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "edge-fallback.step"
    result = generate(
        _INTENT,
        _MATERIAL,
        output,
        config=GenerationConfig(resolution=10, max_iterations=1),
        edge_surrogate=TinyEdgeSurrogate(),
    )
    assert output.is_file()
    assert result["edge_guidance"]["enabled"]
    assert result["edge_guidance"]["authoritative"] is False
    assert result["edge_guidance"]["records"][0]["status"] == (
        "fallback to global level-set evolution"
    )
    assert "untrained" in result["edge_guidance"]["records"][0]["fallback_reason"]


def test_strict_edge_inference_fails_closed_before_export(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "edge-strict.step"
    with pytest.raises(EdgeInferenceError, match="untrained"):
        generate(
            _INTENT,
            _MATERIAL,
            output,
            config=GenerationConfig(resolution=10, max_iterations=1),
            edge_surrogate=TinyEdgeSurrogate(),
            strict_edge_inference=True,
        )
    assert not output.exists()


def test_nonfinite_edge_prediction_falls_back_without_poisoning_generation(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    surrogate = TinyEdgeSurrogate()
    surrogate.mark_trained()
    monkeypatch.setattr(
        surrogate,
        "forward",
        lambda coords, context: torch.full(
            (coords.shape[0], 2), float("nan"), device=coords.device
        ),
    )
    output = tmp_path / "edge-nonfinite.step"
    result = generate(
        _INTENT,
        _MATERIAL,
        output,
        config=GenerationConfig(resolution=10, max_iterations=1),
        edge_surrogate=surrogate,
    )
    assert output.is_file()
    assert all(
        item["status"] == "fallback to global level-set evolution"
        for item in result["edge_guidance"]["records"]
    )
    assert all(
        "non-finite" in item["fallback_reason"]
        for item in result["edge_guidance"]["records"]
    )


def test_macro_governor_vetoes_candidate_edge_stress(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import acwm_orchestrator

    surrogate = TinyEdgeSurrogate()
    surrogate.mark_trained()
    monkeypatch.setattr(
        surrogate,
        "forward",
        lambda coords, context: torch.column_stack(
            (
                torch.zeros(coords.shape[0], device=coords.device),
                torch.full(
                    (coords.shape[0],),
                    2.0 * _MATERIAL.allowable_stress_pa,
                    device=coords.device,
                ),
            )
        ),
    )
    smaller_model = BoxWithThroughBores(
        center=(0.0, 0.0, 0.0),
        half_extents=(0.029, 0.029, 0.03),
    )
    monkeypatch.setattr(
        acwm_orchestrator,
        "fit_box_and_one_through_bore",
        lambda *args, **kwargs: (smaller_model, 0.0),
    )
    output = tmp_path / "edge-veto.step"
    result = generate(
        _INTENT,
        _MATERIAL,
        output,
        config=GenerationConfig(resolution=10, max_iterations=1),
        edge_surrogate=surrogate,
    )
    assert output.is_file()
    assert result["accepted_iterations"] == 0
    assert any(
        item["status"] == "active" for item in result["edge_guidance"]["records"]
    )
    assert any(
        "MacroMicroGovernor rejected proposal" in item["reason"]
        for item in result["history"]
    )


def test_candidate_cannot_relocate_support_and_load_boundary_faces(
    tmp_path, monkeypatch
):
    import acwm_orchestrator

    smaller_model = BoxWithThroughBores(
        center=(0.0, 0.0, 0.0),
        half_extents=(0.029, 0.029, 0.029),
    )
    monkeypatch.setattr(
        acwm_orchestrator,
        "fit_box_and_one_through_bore",
        lambda *args, **kwargs: (smaller_model, 0.0),
    )
    output = tmp_path / "fixed-boundaries.step"
    result = generate(
        _INTENT,
        _MATERIAL,
        output,
        config=GenerationConfig(resolution=10, max_iterations=1),
    )

    assert output.is_file()
    assert result["accepted_iterations"] == 0
    assert result["history"]
    assert all(
        "candidate FEM/topology validation failed" in item["reason"]
        and "does not contain every requested support/load face" in item["reason"]
        for item in result["history"]
    )


def test_cli_runs_intent_to_cad_and_emits_json(tmp_path, capsys):
    pytest.importorskip("cadquery")
    output = tmp_path / "cli-bracket.step"
    assert main(
        [
            "--intent",
            _INTENT,
            "--output",
            str(output),
            "--material-grade",
            _MATERIAL.grade,
            "--young-modulus-pa",
            str(_MATERIAL.young_modulus_pa),
            "--poisson-ratio",
            str(_MATERIAL.poisson_ratio),
            "--allowable-stress-pa",
            str(_MATERIAL.allowable_stress_pa),
            "--resolution",
            "10",
            "--iterations",
            "0",
            "--manufacturing-process",
            "additive",
            "--max-additive-overhang-deg",
            "90",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert output.is_file()
    assert report["output"] == str(output)
    assert report["status"] == "verified exploratory export"
    assert report["manufacturing"]["enabled"]
    assert report["manufacturing"]["checks"] == ["additive_overhang"]


@pytest.mark.parametrize(
    "intent",
    (
        "Steel bracket, downward 10 N.",
        (
            "Steel bracket with 60 x 60 x 60 mm envelope, downward 10 N load "
            "applied to top face and left face, fixed at bottom face."
        ),
        (
            "Steel bracket with 60 x 60 x 60 mm envelope, downward and upward "
            "10 N load applied to top face, fixed at bottom face."
        ),
    ),
)
def test_ambiguous_or_incomplete_constraints_fail_before_export(tmp_path, intent):
    output = tmp_path / "invalid.step"
    with pytest.raises((IntentConstraintError, ValueError)):
        generate(
            intent,
            _MATERIAL,
            output,
            config=GenerationConfig(resolution=10, max_iterations=0),
        )
    assert not output.exists()


def test_output_overwrite_requires_explicit_opt_in(tmp_path):
    output = tmp_path / "existing.step"
    output.write_text("existing output", encoding="ascii")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate(
            _INTENT,
            _MATERIAL,
            output,
            config=GenerationConfig(resolution=10, max_iterations=0),
        )
    assert output.read_text(encoding="ascii") == "existing output"


def test_invalid_material_properties_fail_before_any_side_effect(tmp_path):
    output = tmp_path / "invalid-material.step"
    invalid_material = MaterialProperties(
        grade="",
        young_modulus_pa=200e9,
        poisson_ratio=0.3,
        allowable_stress_pa=1e9,
    )
    with pytest.raises(IntentConstraintError, match="material grade"):
        generate(
            _INTENT,
            invalid_material,
            output,
            config=GenerationConfig(resolution=10, max_iterations=0),
        )
    assert not output.exists()


def test_non_boolean_overwrite_option_is_rejected(tmp_path):
    output = tmp_path / "invalid-config.step"
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        generate(
            _INTENT,
            _MATERIAL,
            output,
            config=GenerationConfig(resolution=10, max_iterations=0, overwrite="false"),
        )
    assert not output.exists()
