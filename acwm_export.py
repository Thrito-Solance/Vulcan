from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from acwm import TriangleMesh, mesh_from_level_set, validate_mesh

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Solid


class BRepExportError(RuntimeError):
    """The level set could not be reconstructed as a valid STEP solid."""


def export_phi_to_step(
    phi: ArrayLike,
    spacing: ArrayLike,
    origin: ArrayLike,
    output_path: str | Path,
    *,
    sewing_tolerance: float | None = None,
) -> Path:
    """Sew a validated level-set triangle surface into one solid and write STEP."""
    destination = Path(output_path)
    if destination.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("level-set B-rep export supports .step and .stp files")
    level_set = np.asarray(phi, dtype=np.float64)
    if not np.all(np.isfinite(level_set)):
        raise ValueError("phi must contain only finite values")
    try:
        mesh = mesh_from_level_set(level_set, spacing, origin)
        validation = validate_mesh(mesh)
    except (ValueError, RuntimeError) as exc:
        raise BRepExportError(f"level-set surface extraction failed: {exc}") from exc
    if not validation.valid:
        raise BRepExportError(
            "refusing to sew a non-manifold triangle surface: "
            + "; ".join(validation.issues)
        )
    if validation.components != 1:
        raise BRepExportError(
            f"STEP export requires one connected boundary component, got {validation.components}"
        )

    extents = np.ptp(mesh.vertices, axis=0)
    scale = float(np.max(extents))
    if not math.isfinite(scale) or scale <= 0:
        raise BRepExportError("surface has no finite positive spatial extent")
    tolerance = (
        max(scale * 1e-9, np.finfo(np.float64).eps * scale * 64)
        if sewing_tolerance is None
        else sewing_tolerance
    )
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("sewing_tolerance must be finite and positive")

    try:
        solid = _build_closed_solid(mesh, tolerance)
        try:
            _write_step(solid, destination)
        finally:
            solid.Nullify()
    except BRepExportError:
        raise
    except Exception as exc:
        raise BRepExportError(
            f"OpenCascade failed during level-set STEP export: {exc}"
        ) from exc
    return destination


def export_step_preview(step_path: str | Path) -> Path:
    """Create an adaptive, fast STL preview from a validated STEP solid."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.StlAPI import StlAPI_Writer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    source_path = Path(step_path)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source_path)) != IFSelect_RetDone:
        raise BRepExportError("OpenCascade could not re-read the exported STEP file")
    if reader.TransferRoots() <= 0:
        raise BRepExportError("OpenCascade found no transferable STEP roots")
    shape = reader.OneShape()
    try:
        if shape.IsNull() or shape.ShapeType() != TopAbs_SOLID:
            raise BRepExportError("exported STEP root is not one solid")
        solids = TopExp_Explorer(shape, TopAbs_SOLID)
        if not solids.More():
            raise BRepExportError("exported STEP contains no solid")
        solids.Next()
        if solids.More() or not BRepCheck_Analyzer(shape, True).IsValid():
            raise BRepExportError("exported STEP did not round-trip as one valid solid")

        bounds = Bnd_Box()
        BRepBndLib.Add_s(shape, bounds, False)
        xmin, ymin, zmin, xmax, ymax, zmax = bounds.Get()
        extents = (xmax - xmin, ymax - ymin, zmax - zmin)
        largest_extent = max(extents)
        if (
            not all(math.isfinite(extent) and extent > 0 for extent in extents)
            or not math.isfinite(largest_extent)
        ):
            raise BRepExportError("exported STEP has invalid bounding dimensions")

        mesher = BRepMesh_IncrementalMesh(
            shape,
            max(largest_extent * 1e-2, 1e-9),
            False,
            0.2,
            True,
        )
        if not mesher.IsDone():
            raise BRepExportError("OpenCascade could not tessellate the STEP preview")
        preview_path = source_path.with_name("model.stl")
        if not StlAPI_Writer().Write(shape, str(preview_path)):
            raise BRepExportError("OpenCascade failed to write the STL preview")
        if (
            not preview_path.is_file()
            or preview_path.stat().st_size == 0
            or preview_path.stat().st_size > 512 * 1024 * 1024
        ):
            raise BRepExportError("OpenCascade produced an invalid or oversized STL preview")
        return preview_path
    finally:
        shape.Nullify()


def _build_closed_solid(mesh: TriangleMesh, tolerance: float) -> TopoDS_Solid:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_Sewing,
        )
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.gp import gp_Pnt
        from OCP.GProp import GProp_GProps
        from OCP.TopAbs import TopAbs_SHELL
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise BRepExportError("OpenCascade OCP bindings are required for STEP export") from exc

    sewing = BRepBuilderAPI_Sewing(tolerance)
    for face_indices in mesh.faces:
        points = mesh.vertices[face_indices]
        polygon = BRepBuilderAPI_MakePolygon(
            *(gp_Pnt(float(x), float(y), float(z)) for x, y, z in points),
            True,
        )
        if not polygon.IsDone():
            raise BRepExportError("OpenCascade failed to construct a triangle boundary")
        face = BRepBuilderAPI_MakeFace(polygon.Wire(), True)
        if not face.IsDone():
            raise BRepExportError("OpenCascade failed to construct a planar triangle face")
        sewing.Add(face.Face())
    sewing.Perform()
    sewed = sewing.SewedShape()
    if sewed.IsNull():
        raise BRepExportError("OpenCascade sewing returned a null shape")

    shells = []
    explorer = TopExp_Explorer(sewed, TopAbs_SHELL)
    while explorer.More():
        shells.append(TopoDS.Shell_s(explorer.Current()))
        explorer.Next()
    if len(shells) != 1:
        raise BRepExportError(
            f"OpenCascade sewing produced {len(shells)} shells; exactly one is required"
        )
    shell = shells[0]
    if not BRep_Tool.IsClosed_s(shell):
        raise BRepExportError("OpenCascade sewing left free shell boundaries")

    solid_builder = BRepBuilderAPI_MakeSolid(shell)
    if not solid_builder.IsDone():
        raise BRepExportError("OpenCascade could not build a solid from the sewn shell")
    solid = solid_builder.Solid()
    if solid.IsNull() or not BRepCheck_Analyzer(solid, True).IsValid():
        raise BRepExportError("OpenCascade produced an invalid B-rep solid")
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, properties)
    volume = float(properties.Mass())
    if not math.isfinite(volume) or volume <= 0:
        raise BRepExportError("OpenCascade solid has non-positive or non-finite volume")
    return solid


def _write_step(solid: TopoDS_Solid, destination: Path) -> None:
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Interface import Interface_Static
        from OCP.STEPControl import (
            STEPControl_AsIs,
            STEPControl_Reader,
            STEPControl_Writer,
        )
        from OCP.TColStd import TColStd_SequenceOfAsciiString
    except ImportError as exc:
        raise BRepExportError("OpenCascade STEP bindings are unavailable") from exc

    writer = STEPControl_Writer()
    if not Interface_Static.SetCVal_s("write.step.unit", "M"):
        raise BRepExportError("OpenCascade could not configure STEP output units to metres")

    expected_properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, expected_properties)
    expected_volume = float(expected_properties.Mass())
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        transfer_status = writer.Transfer(solid, STEPControl_AsIs)
        if transfer_status != IFSelect_RetDone:
            raise BRepExportError(
                f"OpenCascade STEP transfer failed with status {transfer_status}"
            )
        write_status = writer.Write(str(temporary_path))
        if write_status != IFSelect_RetDone:
            raise BRepExportError(
                f"OpenCascade STEP writer failed with status {write_status}"
            )
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise BRepExportError("OpenCascade produced an empty STEP file")

        reader = STEPControl_Reader()
        if reader.ReadFile(str(temporary_path)) != IFSelect_RetDone:
            raise BRepExportError("OpenCascade could not read back the exported STEP file")
        if reader.TransferRoots() <= 0:
            raise BRepExportError("OpenCascade found no transferable STEP roots")
        length_units = TColStd_SequenceOfAsciiString()
        angle_units = TColStd_SequenceOfAsciiString()
        solid_angle_units = TColStd_SequenceOfAsciiString()
        reader.FileUnits(length_units, angle_units, solid_angle_units)
        declared_units = {
            length_units.Value(index).ToCString().lower()
            for index in range(1, length_units.Length() + 1)
        }
        if not declared_units or not declared_units <= {"metre", "meter"}:
            raise BRepExportError(
                f"STEP output must declare metre units, got {sorted(declared_units)}"
            )
        exported_shape = reader.OneShape()
        try:
            exported_properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(exported_shape, exported_properties)
            exported_volume = float(exported_properties.Mass())
            if not math.isfinite(exported_volume) or not math.isclose(
                exported_volume, expected_volume, rel_tol=1e-9, abs_tol=0.0
            ):
                raise BRepExportError(
                    "STEP round-trip volume does not match the source solid "
                    f"({exported_volume:.12g} != {expected_volume:.12g})"
                )
        finally:
            exported_shape.Nullify()
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = ["BRepExportError", "export_phi_to_step", "export_step_preview"]
