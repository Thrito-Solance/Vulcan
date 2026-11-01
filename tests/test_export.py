from __future__ import annotations

import numpy as np
import pytest

from acwm import box_sdf, make_grid, mesh_from_level_set
from acwm_export import BRepExportError, export_phi_to_step


def test_level_set_step_export_reconstructs_one_valid_solid(tmp_path) -> None:
    cq = pytest.importorskip("cadquery")
    points = make_grid((16, 16, 16), (0.1, 0.1, 0.1), (-0.75, -0.75, -0.75))
    phi = box_sdf(points, (0.45, 0.45, 0.45))
    mesh = mesh_from_level_set(phi, (0.1, 0.1, 0.1), (-0.75, -0.75, -0.75))
    triangles = mesh.vertices[mesh.faces]
    expected_volume = abs(
        float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6.0
        )
    )
    path = export_phi_to_step(
        phi,
        (0.1, 0.1, 0.1),
        (-0.75, -0.75, -0.75),
        tmp_path / "level-set.step",
    )

    solid = cq.importers.importStep(str(path)).val()
    assert path.is_file()
    assert solid.isValid()
    assert len(solid.Solids()) == 1
    assert solid.Volume() == pytest.approx(expected_volume, rel=1e-9)

    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TColStd import TColStd_SequenceOfAsciiString

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    length_units = TColStd_SequenceOfAsciiString()
    angle_units = TColStd_SequenceOfAsciiString()
    solid_angle_units = TColStd_SequenceOfAsciiString()
    reader.FileUnits(length_units, angle_units, solid_angle_units)
    units = [
        length_units.Value(index).ToCString()
        for index in range(1, length_units.Length() + 1)
    ]
    assert units == ["metre"]

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(reader.OneShape(), properties)
    assert properties.Mass() == pytest.approx(expected_volume, rel=1e-9)


def test_level_set_step_export_rejects_open_surface_before_creating_output(
    tmp_path,
) -> None:
    coordinates = np.indices((8, 8, 8), dtype=np.float64)
    phi = coordinates[0] - 3.5
    path = tmp_path / "open.step"

    with pytest.raises(BRepExportError, match="non-manifold"):
        export_phi_to_step(phi, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), path)

    assert not path.exists()


def test_level_set_step_export_rejects_disconnected_shells(tmp_path) -> None:
    coordinates = np.indices((20, 20, 20), dtype=np.float64)
    left = sum((coordinates[axis] - 6.0) ** 2 for axis in range(3))
    right = sum((coordinates[axis] - 14.0) ** 2 for axis in range(3))
    phi = np.minimum(left, right) - 16.0
    path = tmp_path / "disconnected.step"

    with pytest.raises(BRepExportError, match="one connected boundary"):
        export_phi_to_step(phi, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), path)

    assert not path.exists()


def test_open_cascade_python_exception_is_wrapped_and_existing_output_is_preserved(
    tmp_path, monkeypatch
) -> None:
    import acwm_export

    points = make_grid((12, 12, 12), (0.1, 0.1, 0.1), (-0.55, -0.55, -0.55))
    phi = box_sdf(points, (0.35, 0.35, 0.35))
    path = tmp_path / "existing.step"
    path.write_text("existing export", encoding="ascii")

    def fail_build(*args, **kwargs):
        raise RuntimeError("simulated OCP failure")

    monkeypatch.setattr(acwm_export, "_build_closed_solid", fail_build)
    with pytest.raises(BRepExportError, match="OpenCascade failed") as error:
        export_phi_to_step(phi, (0.1, 0.1, 0.1), (-0.55, -0.55, -0.55), path)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert path.read_text(encoding="ascii") == "existing export"
