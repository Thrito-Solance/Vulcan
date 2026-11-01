from __future__ import annotations

import json

import numpy as np
import pytest

from acwm import make_grid, sphere_sdf
from acwm_cad import (
    BoxWithThroughBores,
    ThroughBore,
    UnsupportedGeometryError,
    export_step_iges,
    fit_box_and_one_through_bore,
    primitive_sdf,
    verify_primitive_reconstruction,
)
from acwm_intent import (
    intent_to_primitive,
    intent_to_sdf,
    parse_design_intent,
)
from acwm_intent import (
    main as intent_main,
)


def test_analytic_cad_fitting_finds_supported_cylindrical_bore():
    spacing = np.full(3, 0.05)
    origin = np.full(3, -0.5)
    points = make_grid((21, 21, 21), spacing, origin)
    source_model = BoxWithThroughBores(
        center=(0.02, 0.0, 0.0),
        half_extents=(0.4, 0.35, 0.3),
        bores=(ThroughBore("z", (0.07, -0.05), 0.1),),
    )
    phi = primitive_sdf(points, source_model)

    fitted, error = fit_box_and_one_through_bore(
        phi, spacing, origin, tolerance=0.025
    )

    assert error <= 0.025
    assert fitted.bores[0].axis == "z"
    np.testing.assert_allclose(fitted.center, source_model.center, atol=0.01)
    np.testing.assert_allclose(
        fitted.half_extents, source_model.half_extents, atol=0.01
    )
    np.testing.assert_allclose(fitted.bores[0].center, (0.07, -0.05), atol=0.01)
    assert fitted.bores[0].radius == pytest.approx(0.1, abs=0.01)
    assert verify_primitive_reconstruction(
        phi, spacing, origin, fitted, tolerance=0.025
    ) <= 0.025


def test_analytic_cad_fit_rejects_unsupported_geometry():
    spacing = np.full(3, 0.05)
    origin = np.full(3, -0.5)
    points = make_grid((21, 21, 21), spacing, origin)
    phi = sphere_sdf(points, (0.0, 0.0, 0.0), 0.3)
    with pytest.raises(UnsupportedGeometryError, match="outside the supported"):
        fit_box_and_one_through_bore(phi, spacing, origin, tolerance=0.01)


def test_intent_parser_requires_explicit_constraints_before_analysis():
    parsed = parse_design_intent(
        "Design a lightweight titanium motor mount bracket that withstands "
        "a downward 2,000N load and clears a 50mm rotating shaft."
    )
    assert parsed.application == "motor mount bracket"
    assert parsed.material_family == "titanium"
    assert parsed.load_vector_n == (0.0, 0.0, -2000.0)
    assert parsed.shaft_diameter_m == pytest.approx(0.05)
    assert not parsed.ready_for_geometry
    assert not parsed.ready_for_analysis
    assert parsed.shaft_axis is None
    assert "shaft-axis direction" in parsed.missing_requirements
    with pytest.raises(ValueError, match="lacks sufficient geometry"):
        intent_to_primitive(parsed)


def test_explicit_intent_builds_only_a_bounded_analytic_model():
    parsed = parse_design_intent(
        "Titanium motor bracket with 120 x 80 x 40 mm envelope, "
        "50 mm diameter shaft along z-axis, shaft center at (0, 0, 0) mm, "
        "2 mm radial clearance, fixed at left face, downward 2,000 N load "
        "applied to right face."
    )
    assert parsed.bounding_envelope_m == pytest.approx((0.12, 0.08, 0.04))
    assert parsed.shaft_axis == "z"
    assert parsed.shaft_center_m == (0.0, 0.0, 0.0)
    assert parsed.clearance_allowance_m == pytest.approx(0.002)
    assert parsed.load_vector_n == (0.0, 0.0, -2000.0)
    assert parsed.ready_for_geometry
    assert not parsed.ready_for_analysis
    model = intent_to_primitive(parsed)
    assert model.half_extents == pytest.approx((0.06, 0.04, 0.02))
    assert model.bores[0].radius == pytest.approx(0.027)
    assert model.bores[0].center == (0.0, 0.0)
    phi, spacing, origin = intent_to_sdf(parsed, resolution=17)
    assert phi.shape == (17, 17, 17)
    assert spacing == pytest.approx((0.0075, 0.005, 0.0025))
    assert origin == pytest.approx((-0.06, -0.04, -0.02))
    assert phi[8, 8, 8] > 0
    assert phi[0, 8, 8] == pytest.approx(0.0)


def test_intent_parser_rejects_ambiguous_or_invalid_quantities():
    with pytest.raises(ValueError, match="multiple loads"):
        parse_design_intent("Downward 100 N and upward 200 N.")
    with pytest.raises(ValueError, match="direction is ambiguous"):
        parse_design_intent("Bracket for downward and upward 2,000 N.")
    with pytest.raises(ValueError, match="load-application faces are ambiguous"):
        parse_design_intent(
            "Bracket, downward 100 N load applied to left face and right face."
        )
    with pytest.raises(ValueError, match="positive"):
        parse_design_intent("Load downward 0 N.")
    with pytest.raises(ValueError, match="4000-character"):
        parse_design_intent("x" * 4001)
    incomplete_shaft = parse_design_intent("Bracket, 120 x 80 x 40 mm, rotating shaft.")
    assert not incomplete_shaft.ready_for_geometry
    assert "shaft diameter" in incomplete_shaft.missing_requirements


@pytest.mark.parametrize("extension", (".step", ".iges"))
def test_native_cad_exports_round_trip_as_one_solid(tmp_path, extension):
    cq = pytest.importorskip("cadquery")
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.IGESControl import IGESControl_Reader

    model = BoxWithThroughBores(
        center=(0.001, -0.002, 0.003),
        half_extents=(0.04, 0.03, 0.02),
        bores=(ThroughBore("z", (0.005, -0.004), 0.006),),
    )
    path = export_step_iges(model, tmp_path / f"part{extension}")
    assert path.is_file()
    assert path.stat().st_size > 0

    if extension == ".step":
        solid = cq.importers.importStep(str(path)).val()
    else:
        reader = IGESControl_Reader()
        assert reader.ReadFile(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone
        assert reader.TransferRoots() > 0
        solid = cq.Shape.cast(reader.OneShape())

    expected_volume_mm3 = 80 * 60 * 40 - np.pi * 6**2 * 40
    assert solid.isValid()
    assert len(solid.Solids()) == 1
    assert solid.Volume() == pytest.approx(expected_volume_mm3, rel=1e-8)


def test_intent_cli_generates_sdf_and_exports_cad(tmp_path, capsys):
    pytest.importorskip("cadquery")
    output = tmp_path / "mount.step"
    result = intent_main(
        [
            "--resolution",
            "12",
            "--export",
            str(output),
            "--text",
            (
                "Titanium motor bracket with 120 x 80 x 40 mm envelope, "
                "50 mm diameter shaft along z-axis, shaft center at (0, 0, 0) mm, "
                "2 mm radial clearance, fixed at left face, downward 2,000 N load "
                "applied to right face."
            ),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output.is_file()
    assert report["load_vector_n"] == [0.0, 0.0, -2000.0]
    assert report["sdf_grid"]["shape"] == [12, 12, 12]
    assert report["cad_export"]["analysis_ready"] is False


def test_fno_training_is_explicit_grouped_and_checkpointed(tmp_path):
    torch = pytest.importorskip("torch")
    from acwm_surrogate import (
        FNO3d,
        FNOConfig,
        load_trained_fno,
        train_fno,
    )

    config = FNOConfig(
        input_channels=2,
        output_channels=9,
        width=4,
        modes=(2, 2, 2),
        padding=1,
    )
    with pytest.raises(TypeError, match="immutable tuple"):
        FNOConfig(modes=[2, 2, 2]).validate()
    with pytest.raises(ValueError, match="positive integers"):
        FNOConfig(modes=(2, 2.5, 2)).validate()
    untrained = FNO3d(config)
    inputs = torch.rand(4, 2, 8, 8, 8)
    with pytest.raises(RuntimeError, match="untrained FNO"):
        untrained.predict(inputs)
    with pytest.raises(AttributeError):
        untrained.trained = True

    targets = torch.zeros(4, 9, 8, 8, 8)
    checkpoint_path = tmp_path / "model.pt"
    model, history = train_fno(
        inputs,
        targets,
        config=config,
        epochs=1,
        batch_size=1,
        groups=("part-a", "part-a", "part-b", "part-b"),
        checkpoint=checkpoint_path,
    )
    assert model.trained
    assert len(history) == 1
    prediction = model.predict(inputs[:1])
    assert prediction.shape == (1, 9, 8, 8, 8)

    loaded, loaded_history = load_trained_fno(checkpoint_path)
    assert loaded.trained
    assert len(loaded_history) == 1
    assert loaded.predict(inputs[:1]).shape == prediction.shape
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint["training_groups"]).isdisjoint(
        checkpoint["validation_groups"]
    )

    with pytest.raises(ValueError, match="geometry family"):
        train_fno(inputs, targets, config=config, epochs=1)
    with pytest.raises(ValueError, match="floating-point"):
        train_fno(
            torch.ones(4, 2, 8, 8, 8, dtype=torch.int64),
            targets,
            config=config,
            epochs=1,
            groups=("part-a", "part-a", "part-b", "part-b"),
        )

    malformed_checkpoint = tmp_path / "untrained.pt"
    torch.save(
        {
            "format_version": 1,
            "trained": True,
            "config": {
                "input_channels": 2,
                "output_channels": 9,
                "width": 4,
                "modes": [2, 2, 2],
                "padding": 1,
            },
            "state_dict": untrained.state_dict(),
            "history": [],
            "training_groups": [],
            "validation_groups": [],
        },
        malformed_checkpoint,
    )
    with pytest.raises(ValueError, match="training history"):
        load_trained_fno(malformed_checkpoint)
