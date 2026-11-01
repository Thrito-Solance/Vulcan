from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from acwm import (
    FEMResult,
    FloatArray,
    HexMesh,
    MeshValidation,
    TriangleMesh,
    helmholtz_filter,
    ks_stress,
    level_set_step,
    make_grid,
    mesh_from_level_set,
    reinitialize_sdf,
    solve_linear_elasticity,
    validate_mesh,
    voxel_mesh_from_sdf,
)
from acwm_cad import (
    BoxWithThroughBores,
    UnsupportedGeometryError,
    export_step_iges,
    fit_box_and_one_through_bore,
    primitive_sdf,
    verify_primitive_reconstruction,
)
from acwm_edge import MacroContext, MacroMicroGovernor, TinyEdgeSurrogate
from acwm_export import export_phi_to_step
from acwm_intent import ParsedDesignIntent, intent_to_primitive, parse_design_intent
from acwm_manufacturing import (
    ManufacturingConfig,
    ManufacturingGovernor,
    ManufacturingInputError,
    surface_normals_from_sdf,
)

if TYPE_CHECKING:
    from acwm_surrogate import FNO3d


_FACE_AXIS_AND_SIDE: dict[str, tuple[int, bool]] = {
    "left": (0, False),
    "right": (0, True),
    "back": (1, False),
    "front": (1, True),
    "bottom": (2, False),
    "top": (2, True),
}
_MATERIAL_ALLOWABLE_OMISSION = (
    "material grade and verified temperature-dependent allowables"
)


class GenerationError(RuntimeError):
    """The generation request cannot be safely completed."""


class IntentConstraintError(GenerationError):
    """The intent does not specify the required analysis constraints."""


class TopologyInvariantError(GenerationError):
    """A generated boundary mesh violates its required topological invariants."""


class PhysicsConstraintError(GenerationError):
    """The verified finite-element result exceeds the supplied stress limit."""


class SurrogatePreviewError(GenerationError):
    """The optional, non-authoritative surrogate preview cannot be evaluated."""


class EdgeInferenceError(GenerationError):
    """The optional edge surrogate cannot safely provide local guidance."""


@dataclass(frozen=True)
class MaterialProperties:
    grade: str
    young_modulus_pa: float
    poisson_ratio: float
    allowable_stress_pa: float

    def validate(self) -> None:
        if not isinstance(self.grade, str) or not self.grade.strip():
            raise IntentConstraintError("an explicit material grade is required")
        if not math.isfinite(self.young_modulus_pa) or self.young_modulus_pa <= 0:
            raise IntentConstraintError("Young's modulus must be finite and positive")
        if not math.isfinite(self.poisson_ratio) or not (
            -1.0 < self.poisson_ratio < 0.5
        ):
            raise IntentConstraintError(
                "Poisson's ratio must be finite and between -1 and 0.5"
            )
        if (
            not math.isfinite(self.allowable_stress_pa)
            or self.allowable_stress_pa <= 0
        ):
            raise IntentConstraintError(
                "the verified allowable stress must be finite and positive"
            )


@dataclass(frozen=True)
class GenerationConfig:
    resolution: int = 12
    max_iterations: int = 2
    cad_fit_tolerance_m: float | None = None
    overwrite: bool = False
    manufacturing: ManufacturingConfig | None = None

    def validate(self) -> None:
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if (
            not isinstance(self.resolution, int)
            or isinstance(self.resolution, bool)
            or self.resolution < 8
        ):
            raise ValueError("resolution must be an integer of at least 8")
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 0
        ):
            raise ValueError("max_iterations must be a non-negative integer")
        if self.cad_fit_tolerance_m is not None and (
            not math.isfinite(self.cad_fit_tolerance_m)
            or self.cad_fit_tolerance_m <= 0
        ):
            raise ValueError("CAD fit tolerance must be finite and positive")
        if self.manufacturing is not None:
            if not isinstance(self.manufacturing, ManufacturingConfig):
                raise TypeError("manufacturing must be a ManufacturingConfig or None")
            self.manufacturing.validate()


@dataclass(frozen=True)
class _Analysis:
    hex_mesh: HexMesh
    fem: FEMResult
    ks_stress_pa: float
    volume_m3: float


@dataclass(frozen=True)
class _AcceptedState:
    model: BoxWithThroughBores
    phi: FloatArray
    surface: TriangleMesh
    surface_validation: MeshValidation
    analysis: _Analysis


@dataclass(frozen=True)
class _EdgeGuidance:
    sdf_delta: FloatArray
    stress_estimates: FloatArray
    status: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _Preview:
    model: FNO3d

    def evaluate(
        self,
        phi: FloatArray,
        spacing: FloatArray,
        origin: FloatArray,
        load_vector: FloatArray,
        domain_center: FloatArray,
        domain_half_extents: FloatArray,
    ) -> float:
        import torch

        model = self.model
        config = model.config
        if config.input_channels != 8 or config.output_channels != 9:
            raise SurrogatePreviewError(
                "surrogate preview requires 8 input channels and 9 output channels"
            )
        points = make_grid(phi.shape, spacing, origin)
        coordinates = (
            points - domain_center
        ) / domain_half_extents
        direction = load_vector / np.linalg.norm(load_vector)
        features = np.concatenate(
            (
                (phi / float(np.linalg.norm(domain_half_extents)))[..., None],
                (phi <= 0)[..., None].astype(np.float64),
                coordinates,
                np.broadcast_to(direction, (*phi.shape, 3)),
            ),
            axis=-1,
        )
        inputs = torch.from_numpy(
            np.moveaxis(features, -1, 0)[None].astype(np.float32, copy=False)
        )
        if inputs.shape != (1, 8, *phi.shape):
            raise SurrogatePreviewError("surrogate preview feature shape is invalid")
        if any(size < 2 * mode for size, mode in zip(phi.shape, config.modes[:2])):
            raise SurrogatePreviewError(
                "SDF grid is too small for the surrogate's configured Fourier modes"
            )
        if phi.shape[2] // 2 + 1 < config.modes[2]:
            raise SurrogatePreviewError(
                "SDF grid is too small for the surrogate's configured Fourier modes"
            )
        try:
            prediction = model.predict(inputs)
        except (RuntimeError, ValueError) as exc:
            raise SurrogatePreviewError(
                f"surrogate preview failed: {exc}"
            ) from exc
        if prediction.shape != (1, 9, *phi.shape) or not torch.isfinite(
            prediction
        ).all():
            raise SurrogatePreviewError(
                "surrogate preview returned invalid shape or non-finite values"
            )
        return float(prediction.abs().max())


def _grid_for_model(
    model: BoxWithThroughBores,
    resolution: int,
    domain_center: ArrayLike | None = None,
    domain_half_extents: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    center = (
        np.asarray(model.center, dtype=np.float64)
        if domain_center is None
        else np.asarray(domain_center, dtype=np.float64)
    )
    half_extents = (
        np.asarray(model.half_extents, dtype=np.float64)
        if domain_half_extents is None
        else np.asarray(domain_half_extents, dtype=np.float64)
    )
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("domain center must contain three finite coordinates")
    if (
        half_extents.shape != (3,)
        or not np.all(np.isfinite(half_extents))
        or np.any(half_extents <= 0)
    ):
        raise ValueError("domain half-extents must contain three positive values")
    if domain_half_extents is None and model.bores:
        for bore in model.bores:
            axis = "xyz".index(bore.axis)
            transverse = [index for index in range(3) if index != axis]
            half_extents[transverse] += bore.radius * 1.2
    spacing = 2.0 * half_extents / (resolution - 3)
    origin = center - half_extents - spacing
    axes = [
        origin[axis] + np.arange(resolution, dtype=np.float64) * spacing[axis]
        for axis in range(3)
    ]
    for axis in range(3):
        for boundary in (
            center[axis] - model.half_extents[axis],
            center[axis] + model.half_extents[axis],
        ):
            grid_index = (boundary - origin[axis]) / spacing[axis]
            nearest_index = round(float(grid_index))
            if (
                0 <= nearest_index < resolution
                and abs(grid_index - nearest_index) <= 1e-10
            ):
                axes[axis][nearest_index] = boundary
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    phi = primitive_sdf(points, model)
    return phi, spacing, origin


def _face_nodes(
    nodes: FloatArray, face: str, plane_coordinate: float | None = None
) -> np.ndarray:
    axis, is_maximum = _FACE_AXIS_AND_SIDE[face]
    bound = (
        np.max(nodes[:, axis]) if is_maximum else np.min(nodes[:, axis])
    ) if plane_coordinate is None else plane_coordinate
    tolerance = max(
        float(np.max(np.ptp(nodes, axis=0))) * 1e-10,
        np.finfo(np.float64).eps * max(1.0, abs(float(bound))) * 32,
    )
    return np.flatnonzero(np.abs(nodes[:, axis] - bound) <= tolerance)


def _boundary_grid_mask(
    shape: tuple[int, int, int],
    spacing: FloatArray,
    origin: FloatArray,
    boundary_coordinates: dict[str, float],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for face, coordinate in boundary_coordinates.items():
        axis = _FACE_AXIS_AND_SIDE[face][0]
        grid_coordinates = origin[axis] + np.arange(shape[axis]) * spacing[axis]
        indices = np.flatnonzero(
            np.abs(grid_coordinates - coordinate) <= spacing[axis] * 1e-9
        )
        if not len(indices):
            raise IntentConstraintError(
                f"{face} boundary datum is not represented on the SDF grid"
            )
        axis_mask = np.zeros(shape[axis], dtype=bool)
        axis_mask[indices] = True
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        mask |= np.broadcast_to(axis_mask.reshape(reshape), shape)
    return mask


def _analyze(
    model: BoxWithThroughBores,
    phi: FloatArray,
    spacing: FloatArray,
    origin: FloatArray,
    intent: ParsedDesignIntent,
    material: MaterialProperties,
    boundary_coordinates: dict[str, float],
) -> _Analysis:
    hex_mesh = voxel_mesh_from_sdf(phi, spacing, origin)
    support_faces = tuple(
        value.removeprefix("support:")
        for value in intent.supports
        if value.startswith("support:")
    )
    load_faces = tuple(
        value.removeprefix("load:")
        for value in intent.supports
        if value.startswith("load:")
    )
    if not support_faces or len(load_faces) != 1:
        raise IntentConstraintError(
            "analysis requires at least one explicit support face and exactly one load face"
        )
    if any(face not in _FACE_AXIS_AND_SIDE for face in (*support_faces, *load_faces)):
        raise IntentConstraintError("intent contains an unsupported face name")
    if set(support_faces) & set(load_faces):
        raise IntentConstraintError("a face cannot be both supported and loaded")
    fixed_nodes = np.unique(
        np.concatenate(
            [
                _face_nodes(hex_mesh.nodes, face, boundary_coordinates[face])
                for face in support_faces
            ]
        )
    )
    loaded_nodes = _face_nodes(
        hex_mesh.nodes, load_faces[0], boundary_coordinates[load_faces[0]]
    )
    if not len(fixed_nodes) or not len(loaded_nodes):
        raise IntentConstraintError(
            "the voxel mesh does not contain every requested support/load face"
        )
    if np.intersect1d(fixed_nodes, loaded_nodes).size:
        raise IntentConstraintError(
            "support and load faces share constrained nodes; specify a non-overlapping fixture"
        )
    if intent.load_vector_n is None:
        raise IntentConstraintError("a directional load is required")
    load_vector = np.asarray(intent.load_vector_n, dtype=np.float64)
    if load_vector.shape != (3,) or not np.all(np.isfinite(load_vector)):
        raise IntentConstraintError("load vector must contain three finite components")
    if np.linalg.norm(load_vector) <= 0:
        raise IntentConstraintError("load vector must have non-zero magnitude")
    forces = np.zeros_like(hex_mesh.nodes)
    forces[loaded_nodes] = load_vector / len(loaded_nodes)
    fem = solve_linear_elasticity(
        hex_mesh,
        young_modulus=material.young_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=fixed_nodes,
        forces=forces,
        spacing=spacing,
    )
    return _Analysis(
        hex_mesh=hex_mesh,
        fem=fem,
        ks_stress_pa=ks_stress(fem.element_von_mises),
        volume_m3=float(
            np.prod(np.asarray(model.half_extents, dtype=np.float64) * 2.0)
            - sum(
                math.pi * bore.radius**2 * 2.0 * model.half_extents["xyz".index(bore.axis)]
                for bore in model.bores
            )
        ),
    )


def _state_from_model(
    model: BoxWithThroughBores,
    resolution: int,
    intent: ParsedDesignIntent,
    material: MaterialProperties,
    domain_center: FloatArray,
    domain_half_extents: FloatArray,
    boundary_coordinates: dict[str, float],
) -> _AcceptedState:
    phi, spacing, origin = _grid_for_model(
        model, resolution, domain_center, domain_half_extents
    )
    surface = mesh_from_level_set(phi, spacing, origin)
    validation = validate_mesh(surface)
    if not validation.valid:
        raise TopologyInvariantError(
            "extracted surface violates topology invariants: "
            + ", ".join(validation.issues)
        )
    verify_primitive_reconstruction(
        phi,
        spacing,
        origin,
        model,
        tolerance=np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(phi))))
        * 32,
    )
    analysis = _analyze(
        model, phi, spacing, origin, intent, material, boundary_coordinates
    )
    return _AcceptedState(model, phi, surface, validation, analysis)


def _manufacturing_normals(
    state: _AcceptedState,
    spacing: FloatArray,
    origin: FloatArray,
    config: ManufacturingConfig | None,
) -> FloatArray | None:
    if config is None:
        return None
    try:
        return surface_normals_from_sdf(
            state.phi,
            spacing,
            origin,
            state.surface,
            smoothing_sigma_cells=config.smoothing_sigma_cells,
            minimum_gradient_norm=config.minimum_gradient_norm,
        )
    except ManufacturingInputError as exc:
        raise GenerationError(
            f"manufacturing normal estimation failed closed: {exc}"
        ) from exc


def _stress_velocity(
    state: _AcceptedState, spacing: FloatArray, origin: FloatArray, allowable: float
) -> FloatArray:
    stress_sum = np.zeros(state.phi.shape, dtype=np.float64)
    stress_count = np.zeros(state.phi.shape, dtype=np.int64)
    for element, stress in zip(
        state.analysis.hex_mesh.elements,
        state.analysis.fem.element_von_mises,
        strict=True,
    ):
        cell_origin = np.min(state.analysis.hex_mesh.nodes[element], axis=0)
        cell = np.rint((cell_origin - origin) / spacing).astype(np.int64)
        for offset in (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ):
            index = tuple(cell + offset)
            stress_sum[index] += stress
            stress_count[index] += 1
    represented = stress_count > 0
    stress_grid = np.full(state.phi.shape, allowable, dtype=np.float64)
    stress_grid[represented] = stress_sum[represented] / stress_count[represented]
    utilization = np.clip(stress_grid / allowable, 0.0, 1.0)
    return -np.where(state.phi <= 0, 1.0 - utilization, 0.0)


def _macro_context(
    state: _AcceptedState, domain_half_extents: FloatArray, allowable: float
) -> MacroContext:
    displacement = state.analysis.fem.displacement
    maximum_displacement = float(np.max(np.linalg.norm(displacement, axis=1)))
    characteristic_length = float(np.linalg.norm(2.0 * domain_half_extents))
    return MacroContext(
        global_compliance_error=maximum_displacement / characteristic_length,
        stress_scaling_factor=state.analysis.ks_stress_pa / allowable,
        carving_aggressiveness=float(
            np.clip(1.0 - state.analysis.ks_stress_pa / allowable, 0.05, 1.0)
        ),
    )


def _edge_guidance(
    surrogate: TinyEdgeSurrogate | None,
    state: _AcceptedState,
    spacing: FloatArray,
    origin: FloatArray,
    domain_center: FloatArray,
    domain_half_extents: FloatArray,
    allowable: float,
    *,
    strict: bool,
) -> _EdgeGuidance:
    if surrogate is None:
        return _EdgeGuidance(
            np.zeros(state.phi.shape, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            "disabled",
        )
    try:
        if not surrogate.is_trained:
            raise RuntimeError("edge surrogate is untrained")
        import torch

        parameters = tuple(surrogate.parameters())
        if not parameters:
            raise ValueError("edge surrogate has no model parameters")
        points = make_grid(state.phi.shape, spacing, origin)
        coordinates = (
            (points - domain_center) / domain_half_extents
        ).reshape(-1, 3)
        coordinates_tensor = torch.as_tensor(
            coordinates,
            dtype=parameters[0].dtype,
            device=parameters[0].device,
        )
        context = _macro_context(state, domain_half_extents, allowable)
        with torch.no_grad():
            prediction = surrogate(coordinates_tensor, context)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("edge surrogate output must be a tensor")
        if prediction.shape != (coordinates.shape[0], 2):
            raise ValueError(
                "edge surrogate output must have shape (N, 2), "
                f"got {tuple(prediction.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise ValueError("edge surrogate returned non-finite values")
        output = prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if np.any(output[:, 1] < 0):
            raise ValueError("edge surrogate returned negative stress estimates")
        return _EdgeGuidance(
            sdf_delta=output[:, 0].reshape(state.phi.shape),
            stress_estimates=output[:, 1].copy(),
            status="active",
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        if strict:
            raise EdgeInferenceError(f"edge inference failed closed: {exc}") from exc
        return _EdgeGuidance(
            np.zeros(state.phi.shape, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            "fallback to global level-set evolution",
            str(exc),
        )


def _load_preview(path: Path | None) -> _Preview | None:
    if path is None:
        return None
    from acwm_surrogate import load_trained_fno

    model, _ = load_trained_fno(path)
    if model.config.input_channels != 8 or model.config.output_channels != 9:
        raise SurrogatePreviewError(
            "surrogate checkpoint must use 8 input and 9 output channels"
        )
    return _Preview(model)


def _record_preview(
    preview: _Preview | None,
    state: _AcceptedState,
    spacing: FloatArray,
    origin: FloatArray,
    intent: ParsedDesignIntent,
    domain_center: FloatArray,
    domain_half_extents: FloatArray,
) -> float | None:
    if preview is None:
        return None
    if intent.load_vector_n is None:
        raise IntentConstraintError("a directional load is required for surrogate preview")
    return preview.evaluate(
        state.phi,
        spacing,
        origin,
        np.asarray(intent.load_vector_n, dtype=np.float64),
        domain_center,
        domain_half_extents,
    )


def validate_generation_request(
    intent_text: str,
    material: MaterialProperties,
    config: GenerationConfig | None = None,
) -> tuple[ParsedDesignIntent, BoxWithThroughBores]:
    """Validate bounded design inputs before callers start expensive work."""
    if config is None:
        config = GenerationConfig()
    config.validate()
    material.validate()
    intent = parse_design_intent(intent_text)
    if not intent.ready_for_geometry:
        raise IntentConstraintError(
            "intent lacks sufficient geometry constraints: "
            + ", ".join(intent.missing_requirements)
        )
    if intent.load_vector_n is None:
        raise IntentConstraintError("intent must specify one directional force")
    if intent.material_family is None:
        raise IntentConstraintError("intent must specify a material family")
    if (
        _MATERIAL_ALLOWABLE_OMISSION not in intent.missing_requirements
        and not intent.ready_for_analysis
    ):
        raise IntentConstraintError(
            "intent contains unresolved analysis requirements: "
            + ", ".join(intent.missing_requirements)
        )
    return intent, intent_to_primitive(intent)


def generate(
    intent_text: str,
    material: MaterialProperties,
    output_path: str | Path,
    *,
    config: GenerationConfig | None = None,
    surrogate_checkpoint: str | Path | None = None,
    edge_surrogate: TinyEdgeSurrogate | None = None,
    strict_edge_inference: bool = False,
) -> dict[str, object]:
    if config is None:
        config = GenerationConfig()
    intent, model = validate_generation_request(intent_text, material, config)
    if not isinstance(strict_edge_inference, bool):
        raise TypeError("strict_edge_inference must be a boolean")
    if edge_surrogate is not None and not isinstance(
        edge_surrogate, TinyEdgeSurrogate
    ):
        raise TypeError("edge_surrogate must be a TinyEdgeSurrogate or None")
    destination = Path(output_path)
    if destination.suffix.lower() not in {".step", ".stp", ".iges", ".igs"}:
        raise ValueError("output must use a .step, .stp, .iges, or .igs extension")
    if destination.exists() and not config.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing CAD output: {destination}"
        )

    model_center = np.asarray(model.center, dtype=np.float64)
    model_half_extents = np.asarray(model.half_extents, dtype=np.float64)
    boundary_coordinates = {
        face: float(
            model_center[axis] + (model_half_extents[axis] if is_maximum else -model_half_extents[axis])
        )
        for face, (axis, is_maximum) in _FACE_AXIS_AND_SIDE.items()
    }
    preview = _load_preview(
        Path(surrogate_checkpoint) if surrogate_checkpoint is not None else None
    )

    output_half_extents = np.asarray(model.half_extents, dtype=np.float64)
    if model.bores:
        for bore in model.bores:
            axis = "xyz".index(bore.axis)
            transverse = [index for index in range(3) if index != axis]
            output_half_extents[transverse] = np.maximum(
                output_half_extents[transverse],
                output_half_extents[transverse] + bore.radius * 1.2,
            )
    output_spacing = 2.0 * output_half_extents / (config.resolution - 3)
    output_origin = np.asarray(model.center) - output_half_extents - output_spacing
    fit_tolerance = (
        config.cad_fit_tolerance_m
        if config.cad_fit_tolerance_m is not None
        else float(np.min(output_spacing)) * 0.05
    )
    if fit_tolerance > float(np.min(output_spacing)) * 0.5:
        raise ValueError("CAD fit tolerance must not exceed half the smallest grid spacing")

    state = _state_from_model(
        model,
        config.resolution,
        intent,
        material,
        model_center,
        output_half_extents,
        boundary_coordinates,
    )
    manufacturing_governor = (
        ManufacturingGovernor(config.manufacturing)
        if config.manufacturing is not None
        else None
    )
    governor = MacroMicroGovernor(
        material.allowable_stress_pa, manufacturing_governor
    )
    if state.analysis.ks_stress_pa > material.allowable_stress_pa:
        raise PhysicsConstraintError(
            "initial design exceeds the supplied allowable stress "
            f"({state.analysis.ks_stress_pa:.6g} Pa > "
            f"{material.allowable_stress_pa:.6g} Pa); no CAD file was written"
        )
    initial_governor_accepted, initial_governor_reason = (
        governor.evaluate_and_filter(
            state.analysis.fem.element_von_mises,
            state.surface_validation.valid,
            _manufacturing_normals(
                state, output_spacing, output_origin, config.manufacturing
            ),
        )
    )
    if not initial_governor_accepted:
        raise PhysicsConstraintError(
            f"initial design rejected by MacroMicroGovernor: {initial_governor_reason}"
        )
    preview_records: list[float] = []
    initial_preview = _record_preview(
        preview,
        state,
        output_spacing,
        output_origin,
        intent,
        np.asarray(model.center, dtype=np.float64),
        output_half_extents,
    )
    if initial_preview is not None:
        preview_records.append(initial_preview)

    history: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []
    stop_reason = "iteration limit reached"
    spacing = output_spacing
    origin = output_origin
    requested_faces = tuple(
        value.removeprefix("support:")
        for value in intent.supports
        if value.startswith("support:")
    ) + tuple(
        value.removeprefix("load:")
        for value in intent.supports
        if value.startswith("load:")
    )
    active_boundary_coordinates = {
        face: boundary_coordinates[face] for face in requested_faces
    }
    fixed_boundary_mask = _boundary_grid_mask(
        state.phi.shape, spacing, origin, active_boundary_coordinates
    )
    for iteration in range(1, config.max_iterations + 1):
        edge = _edge_guidance(
            edge_surrogate,
            state,
            spacing,
            origin,
            np.asarray(model.center, dtype=np.float64),
            output_half_extents,
            material.allowable_stress_pa,
            strict=strict_edge_inference,
        )
        edge_records.append(
            {
                "iteration": iteration,
                "status": edge.status,
                "fallback_reason": edge.fallback_reason,
            }
        )
        edge_context = _macro_context(
            state, output_half_extents, material.allowable_stress_pa
        )
        edge_delta_limit = (
            float(np.min(spacing)) * 0.1 * edge_context.carving_aggressiveness
        )
        bounded_edge_delta = np.clip(
            edge.sdf_delta, 0.0, edge_delta_limit
        )
        bounded_edge_delta[fixed_boundary_mask] = 0.0
        velocity = _stress_velocity(
            state, spacing, origin, material.allowable_stress_pa
        )
        filtered_velocity = helmholtz_filter(
            velocity, radius=float(np.min(spacing)), spacing=spacing
        )
        filtered_velocity[fixed_boundary_mask] = 0.0
        dt = float(np.min(spacing)) * 0.2
        accepted = False
        for backtrack in range(4):
            trial_dt = dt / (2**backtrack)
            trial_phi = level_set_step(
                state.phi, filtered_velocity, trial_dt, spacing
            )
            trial_phi = trial_phi + np.where(
                state.phi <= 0, bounded_edge_delta, 0.0
            )
            if not np.any(trial_phi <= 0) or np.all(trial_phi <= 0):
                _, governor_reason = governor.evaluate_and_filter(
                    state.analysis.fem.element_von_mises, False
                )
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": (
                            "level-set proposal removed the domain or lost its exterior; "
                            + governor_reason
                        ),
                    }
                )
                stop_reason = "level-set proposal violated the signed-domain invariant"
                break
            try:
                reinitialized = reinitialize_sdf(trial_phi, spacing)
            except ValueError as exc:
                _, governor_reason = governor.evaluate_and_filter(
                    state.analysis.fem.element_von_mises, False
                )
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": f"{exc}; {governor_reason}",
                    }
                )
                stop_reason = "level-set proposal could not be reinitialized"
                break
            trial_surface = mesh_from_level_set(reinitialized, spacing, origin)
            trial_validation = validate_mesh(trial_surface)
            if not trial_validation.valid:
                _, governor_reason = governor.evaluate_and_filter(
                    state.analysis.fem.element_von_mises, False
                )
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": (
                            "; ".join(trial_validation.issues)
                            + f"; {governor_reason}"
                        ),
                    }
                )
                stop_reason = "level-set proposal violated mesh topology invariants"
                break
            try:
                proposed_model, fit_error = fit_box_and_one_through_bore(
                    reinitialized,
                    spacing,
                    origin,
                    tolerance=fit_tolerance,
                )
            except UnsupportedGeometryError as exc:
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": str(exc),
                    }
                )
                continue
            proposed_phi, _, _ = _grid_for_model(
                proposed_model,
                config.resolution,
                np.asarray(model.center, dtype=np.float64),
                output_half_extents,
            )
            if proposed_phi.shape != state.phi.shape:
                raise GenerationError(
                    "primitive projection changed the configured spatial grid"
                )
            if np.any((proposed_phi <= 0) & (state.phi > 0)):
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": "primitive projection would add material outside the accepted design",
                    }
                )
                continue
            if np.count_nonzero(proposed_phi <= 0) >= np.count_nonzero(state.phi <= 0):
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": "primitive projection did not remove resolved grid material",
                    }
                )
                stop_reason = "no resolvable material reduction at the configured grid spacing"
                break
            try:
                proposed = _state_from_model(
                    proposed_model,
                    config.resolution,
                    intent,
                    material,
                    model_center,
                    output_half_extents,
                    boundary_coordinates,
                )
            except (GenerationError, ValueError, RuntimeError) as exc:
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": f"candidate FEM/topology validation failed: {exc}",
                    }
                )
                continue
            candidate_edge = _edge_guidance(
                edge_surrogate,
                proposed,
                spacing,
                origin,
                np.asarray(model.center, dtype=np.float64),
                output_half_extents,
                material.allowable_stress_pa,
                strict=strict_edge_inference,
            )
            edge_records.append(
                {
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "status": candidate_edge.status,
                    "fallback_reason": candidate_edge.fallback_reason,
                }
            )
            governor_stress = proposed.analysis.fem.element_von_mises
            if candidate_edge.stress_estimates.size:
                governor_stress = np.concatenate(
                    (governor_stress, candidate_edge.stress_estimates)
                )
            candidate_normals = _manufacturing_normals(
                proposed, spacing, origin, config.manufacturing
            )
            governor_accepted, governor_reason = governor.evaluate_and_filter(
                governor_stress,
                proposed.surface_validation.valid,
                candidate_normals,
            )
            if not governor_accepted:
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": f"MacroMicroGovernor rejected proposal: {governor_reason}",
                    }
                )
                continue
            if proposed.analysis.ks_stress_pa > material.allowable_stress_pa:
                history.append(
                    {
                        "iteration": iteration,
                        "backtrack": backtrack,
                        "accepted": False,
                        "status": "rollback",
                        "reason": (
                            "FEA verification exceeded the supplied allowable stress "
                            f"({proposed.analysis.ks_stress_pa:.6g} Pa > "
                            f"{material.allowable_stress_pa:.6g} Pa)"
                        ),
                        "ks_stress_pa": proposed.analysis.ks_stress_pa,
                    }
                )
                continue
            preview_value = _record_preview(
                preview,
                proposed,
                spacing,
                origin,
                intent,
                np.asarray(model.center, dtype=np.float64),
                output_half_extents,
            )
            if preview_value is not None:
                preview_records.append(preview_value)
            history.append(
                {
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "accepted": True,
                    "status": "accepted",
                    "max_sdf_fit_error_m": fit_error,
                    "volume_before_m3": state.analysis.volume_m3,
                    "volume_after_m3": proposed.analysis.volume_m3,
                    "ks_stress_pa": proposed.analysis.ks_stress_pa,
                    "surrogate_preview_max_abs_raw": preview_value,
                }
            )
            state = proposed
            accepted = True
            break
        if stop_reason.startswith(("level-set proposal", "no resolvable")):
            break
        if not accepted:
            stop_reason = "all bounded backtracking proposals were rejected; last valid state retained"
            break
    else:
        if config.max_iterations == 0:
            stop_reason = "carving disabled by zero iteration limit"

    output_phi, output_spacing, output_origin = _grid_for_model(
        state.model,
        config.resolution,
        np.asarray(model.center, dtype=np.float64),
        output_half_extents,
    )
    verify_primitive_reconstruction(
        output_phi,
        output_spacing,
        output_origin,
        state.model,
        tolerance=np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(output_phi))))
        * 32,
    )
    final_mesh = mesh_from_level_set(output_phi, output_spacing, output_origin)
    final_validation = validate_mesh(final_mesh)
    if not final_validation.valid:
        raise TopologyInvariantError(
            "final CAD mesh failed topology validation: "
            + ", ".join(final_validation.issues)
        )
    final_triangles = final_mesh.vertices[final_mesh.faces]
    final_volume_m3 = abs(
        float(
            np.einsum(
                "ij,ij->i",
                final_triangles[:, 0],
                np.cross(final_triangles[:, 1], final_triangles[:, 2]),
            ).sum()
            / 6.0
        )
    )
    if not math.isfinite(final_volume_m3) or final_volume_m3 <= 0:
        raise TopologyInvariantError("final CAD mesh has invalid enclosed volume")
    if state.analysis.ks_stress_pa > material.allowable_stress_pa:
        raise PhysicsConstraintError("refusing CAD export for an over-limit design")
    final_governor_accepted, final_governor_reason = governor.evaluate_and_filter(
        state.analysis.fem.element_von_mises,
        final_validation.valid,
        _manufacturing_normals(
            state, output_spacing, output_origin, config.manufacturing
        ),
    )
    if not final_governor_accepted:
        raise PhysicsConstraintError(
            f"final design rejected before CAD export: {final_governor_reason}"
        )
    if destination.suffix.lower() in {".step", ".stp"}:
        exported = export_phi_to_step(
            output_phi, output_spacing, output_origin, destination
        )
    else:
        exported = export_step_iges(state.model, destination)
    return {
        "status": "verified exploratory export",
        "output": str(exported),
        "material_grade": material.grade,
        "allowable_stress_pa": material.allowable_stress_pa,
        "resolution": config.resolution,
        "max_iterations": config.max_iterations,
        "accepted_iterations": sum(bool(item["accepted"]) for item in history),
        "stop_reason": stop_reason,
        "final_volume_m3": final_volume_m3,
        "maximum_sampled_von_mises_pa": state.analysis.fem.maximum_von_mises,
        "ks_von_mises_pa": state.analysis.ks_stress_pa,
        "surface_validation": asdict(final_validation),
        "surface_faces": len(final_mesh.faces),
        "manufacturing": {
            "enabled": config.manufacturing is not None,
            "checks": (
                []
                if config.manufacturing is None
                else [
                    name
                    for name, enabled in (
                        ("cnc_directional_normal", config.manufacturing.check_cnc),
                        ("additive_overhang", config.manufacturing.check_additive),
                    )
                    if enabled
                ]
            ),
            "passed": True,
            "qualification": "heuristic screen only; not a toolpath or process certification",
        },
        "surrogate_preview": {
            "enabled": preview is not None,
            "max_abs_raw_outputs": preview_records,
            "authoritative": False,
        },
        "edge_guidance": {
            "enabled": edge_surrogate is not None,
            "strict_inference": strict_edge_inference,
            "authoritative": False,
            "veto_only_stress_input": True,
            "records": edge_records,
        },
        "history": history,
        "limitations": [
            "coarse voxel FEA is exploratory, not engineering certification",
            "surrogate outputs are uncalibrated previews and never gate acceptance",
            "native CAD is limited to one valid box with at most one straight through-bore",
            "external material grade and allowable values are caller-supplied and unverified",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded ACWM level-set proposals with FEA verification"
    )
    parser.add_argument("--intent", required=True, help="Natural-language design intent")
    parser.add_argument("--output", required=True, type=Path, help="STEP or IGES output path")
    parser.add_argument("--material-grade", required=True)
    parser.add_argument("--young-modulus-pa", required=True, type=float)
    parser.add_argument("--poisson-ratio", required=True, type=float)
    parser.add_argument("--allowable-stress-pa", required=True, type=float)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--cad-fit-tolerance-m", type=float)
    parser.add_argument("--surrogate-checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manufacturing-process",
        choices=("none", "cnc", "additive", "both"),
        default="none",
        help="enable optional directional-normal manufacturing screens",
    )
    parser.add_argument(
        "--tool-pull-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--max-undercut-tolerance", type=float, default=0.01)
    parser.add_argument("--max-additive-overhang-deg", type=float, default=45.0)
    arguments = parser.parse_args(argv)
    manufacturing = None
    if arguments.manufacturing_process != "none":
        manufacturing = ManufacturingConfig(
            tool_pull_axis=tuple(arguments.tool_pull_axis),
            max_undercut_tolerance=arguments.max_undercut_tolerance,
            max_additive_overhang_deg=arguments.max_additive_overhang_deg,
            check_cnc=arguments.manufacturing_process in {"cnc", "both"},
            check_additive=arguments.manufacturing_process in {"additive", "both"},
        )
    result = generate(
        arguments.intent,
        MaterialProperties(
            grade=arguments.material_grade,
            young_modulus_pa=arguments.young_modulus_pa,
            poisson_ratio=arguments.poisson_ratio,
            allowable_stress_pa=arguments.allowable_stress_pa,
        ),
        arguments.output,
        config=GenerationConfig(
            resolution=arguments.resolution,
            max_iterations=arguments.iterations,
            cad_fit_tolerance_m=arguments.cad_fit_tolerance_m,
            overwrite=arguments.overwrite,
            manufacturing=manufacturing,
        ),
        surrogate_checkpoint=arguments.surrogate_checkpoint,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
