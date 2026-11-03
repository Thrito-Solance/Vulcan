"""Export bridge with separate analytic and bounded level-set guarantees."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acwm import mesh_from_level_set, validate_mesh
from acwm_cad import export_step_iges
from acwm_export import BRepExportError, export_phi_to_step

from .errors import ExportError
from .features import LevelSetField
from .schema import (
    ExportPath,
    FeatureEffect,
    FeatureRequest,
    Fidelity,
    TranslationReport,
)
from .translator import translate


@dataclass(frozen=True)
class ExportResult:
    path: ExportPath
    output: str
    primitives: tuple[str, ...]
    guarantee: str
    warnings: tuple[str, ...]
    fidelity: Fidelity
    landed: tuple[FeatureEffect, ...]
    translation: TranslationReport


def export(
    request: FeatureRequest,
    output: Path,
    *,
    path: ExportPath,
    resolution: int = 32,
) -> ExportResult:
    kind, payload, report = translate(request, path=path, resolution=resolution)
    output = Path(output)
    if kind == "analytic":
        try:
            written = export_step_iges(payload, output)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise ExportError(f"analytic export failed: {error}") from error
        return ExportResult(
            path=ExportPath.ANALYTIC,
            output=str(written),
            primitives=report.primitives,
            guarantee="verified analytic primitive (OCP round-trip, one closed solid)",
            warnings=report.warnings,
            fidelity=Fidelity(mode="exact"),
            landed=(),
            translation=report,
        )

    field: LevelSetField = payload
    try:
        mesh = mesh_from_level_set(field.phi, field.spacing, field.origin)
        validation = validate_mesh(mesh)
    except (RuntimeError, ValueError) as error:
        raise ExportError(f"level-set mesh validation failed: {error}") from error
    if not validation.valid:
        raise ExportError(f"level-set mesh invalid: {validation.issues}")
    try:
        written = export_phi_to_step(
            field.phi, field.spacing, field.origin, output
        )
    except (BRepExportError, OSError, RuntimeError, ValueError) as error:
        raise ExportError(f"level-set export failed: {error}") from error
    return ExportResult(
        path=ExportPath.LEVEL_SET,
        output=str(written),
        primitives=report.primitives,
        guarantee=(
            "watertight faceted B-rep; NOT an analytic primitive and NOT "
            "verified by the FEA loop"
        ),
        warnings=report.warnings,
        fidelity=field.fidelity,
        landed=field.landed,
        translation=report,
    )
