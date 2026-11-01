from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from acwm_cad import Axis, BoxWithThroughBores, ThroughBore

_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_SIGNED_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_UNIT_TO_METERS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
_DIRECTION_TO_VECTOR = {
    "down": (0.0, 0.0, -1.0),
    "downward": (0.0, 0.0, -1.0),
    "up": (0.0, 0.0, 1.0),
    "upward": (0.0, 0.0, 1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "forward": (0.0, 1.0, 0.0),
    "backward": (0.0, -1.0, 0.0),
}


@dataclass(frozen=True)
class ParsedDesignIntent:
    application: str | None
    material_family: str | None
    material_grade: str | None
    bounding_envelope_m: tuple[float, float, float] | None
    load_vector_n: tuple[float, float, float] | None
    shaft_diameter_m: float | None
    shaft_axis: str | None
    shaft_center_m: tuple[float, float, float] | None
    clearance_allowance_m: float | None
    supports: tuple[str, ...]
    ready_for_geometry: bool
    ready_for_analysis: bool
    missing_requirements: tuple[str, ...]
    assumptions: tuple[str, ...]
    parser: str = "local deterministic grammar; no external model or network"


def _number(value: str) -> float:
    number = float(value.replace(",", ""))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("parsed dimensions and loads must be positive")
    return number


def _coordinate(value: str) -> float:
    coordinate = float(value.replace(",", ""))
    if not math.isfinite(coordinate):
        raise ValueError("parsed coordinates must be finite")
    return coordinate


def _parse_envelope(text: str) -> tuple[float, float, float] | None:
    pattern = re.compile(
        rf"(?P<a>{_NUMBER})\s*(?P<u1>mm|cm|m)?\s*[x×]\s*"
        rf"(?P<b>{_NUMBER})\s*(?P<u2>mm|cm|m)?\s*[x×]\s*"
        rf"(?P<c>{_NUMBER})\s*(?P<u3>mm|cm|m)?\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError("multiple bounding envelopes are ambiguous")
    if not matches:
        return None
    match = matches[0]
    unit_groups = [match.group(label) for label in ("u1", "u2", "u3")]
    default_unit = next((unit for unit in reversed(unit_groups) if unit), None)
    if default_unit is None:
        return None
    values = []
    for label, unit_label in zip(("a", "b", "c"), ("u1", "u2", "u3")):
        size = _number(match.group(label))
        dimension_unit = (match.group(unit_label) or default_unit).lower()
        values.append(size * _UNIT_TO_METERS[dimension_unit])
    return tuple(values)


def _parse_load(text: str) -> tuple[float, float, float] | None:
    magnitude_pattern = re.compile(
        rf"\b(?P<magnitude>{_NUMBER})\s*(?P<unit>kN|N)\b",
        re.IGNORECASE,
    )
    direction_pattern = re.compile(
        r"\b(?P<direction>downward|down|upward|up|left|right|forward|backward)\b",
        re.IGNORECASE,
    )
    magnitudes = list(magnitude_pattern.finditer(text))
    if len(magnitudes) > 1:
        raise ValueError("multiple loads are ambiguous; provide one load per design intent")
    if not magnitudes:
        return None
    magnitude_match = magnitudes[0]
    sentence_start = max(
        text.rfind(".", 0, magnitude_match.start()),
        text.rfind(";", 0, magnitude_match.start()),
        text.rfind("\n", 0, magnitude_match.start()),
    ) + 1
    preceding_directions = [
        match
        for match in direction_pattern.finditer(
            text, sentence_start, magnitude_match.start()
        )
        if not re.match(
            r"\s*[- ]?\s*(?:face|side|edge)\b",
            text[match.end() :],
            re.IGNORECASE,
        )
    ]
    if not preceding_directions:
        return None
    vectors = {
        _DIRECTION_TO_VECTOR[match.group("direction").lower()]
        for match in preceding_directions
    }
    if len(vectors) > 1:
        raise ValueError("load direction is ambiguous")
    direction = preceding_directions[-1].group("direction").lower()
    magnitude = _number(magnitude_match.group("magnitude"))
    if magnitude_match.group("unit").lower() == "kn":
        magnitude *= 1000.0
    unit_vector = _DIRECTION_TO_VECTOR[direction]
    return tuple(magnitude * component for component in unit_vector)


def _parse_shaft_diameter(text: str) -> float | None:
    patterns = (
        re.compile(
            rf"\b(?P<size>{_NUMBER})\s*(?P<unit>mm|cm|m)\s*"
            r"(?:(?:diameter|diam\.?|rotating\s+shaft|shaft)\b)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:shaft\s+)?(?:diameter|diam\.?)\s*(?:of\s*)?"
            rf"(?P<size>{_NUMBER})\s*(?P<unit>mm|cm|m)\b",
            re.IGNORECASE,
        ),
    )
    matches = [
        match
        for pattern in patterns
        for match in pattern.finditer(text)
    ]
    distinct_matches = {
        (
            float(match.group("size").replace(",", "")),
            match.group("unit").lower(),
        ): match
        for match in matches
    }
    if len(distinct_matches) > 1:
        raise ValueError("multiple shaft sizes are ambiguous")
    if not distinct_matches:
        return None
    match = next(iter(distinct_matches.values()))
    return _number(match.group("size")) * _UNIT_TO_METERS[match.group("unit").lower()]


def _parse_shaft_axis(text: str) -> str | None:
    pattern = re.compile(
        r"\b(?:shaft\s+axis|axis\s+of\s+(?:the\s+)?shaft)\s*"
        r"(?:is\s*)?(?:along\s*)?(?P<axis>[xyz])(?:\s*[- ]?axis)?\b"
        r"|\bshaft\s+(?:along|parallel\s+to)\s+(?:the\s+)?"
        r"(?P<direction>[xyz])(?:\s*[- ]?axis)?\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError("multiple shaft axes are ambiguous")
    if not matches:
        return None
    match = matches[0]
    return (match.group("axis") or match.group("direction")).lower()


def _parse_shaft_center(text: str) -> tuple[float, float, float] | None:
    pattern = re.compile(
        rf"\bshaft\s+center(?:\s+at|\s*[:=])?\s*"
        rf"\(\s*(?P<x>{_SIGNED_NUMBER})\s*,\s*(?P<y>{_SIGNED_NUMBER})\s*,\s*"
        rf"(?P<z>{_SIGNED_NUMBER})\s*\)\s*(?P<unit>mm|cm|m)\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError("multiple shaft centers are ambiguous")
    if not matches:
        return None
    match = matches[0]
    scale = _UNIT_TO_METERS[match.group("unit").lower()]
    return tuple(_coordinate(match.group(axis)) * scale for axis in ("x", "y", "z"))


def _parse_clearance(text: str) -> float | None:
    pattern = re.compile(
        rf"\b(?P<size>{_NUMBER})\s*(?P<unit>mm|cm|m)\s*"
        r"(?:(?:radial|minimum)\s+)?clearance\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError("multiple clearance allowances are ambiguous")
    if not matches:
        return None
    match = matches[0]
    return _number(match.group("size")) * _UNIT_TO_METERS[match.group("unit").lower()]


def _parse_faces(text: str, action: str) -> tuple[str, ...]:
    if action == "support":
        pattern = re.compile(
            r"\b(?:fixed|support(?:ed)?|mount(?:ed)?|bolt(?:ed)?)\s+"
            r"(?:at|on|to)\s+"
            r"(?P<face>left|right|top|bottom|front|back)(?:\s+face)?\b",
            re.IGNORECASE,
        )
    elif action == "load":
        pattern = re.compile(
            r"\bload\s+(?:applied\s+)?(?:at|to|on)\s+"
            r"(?:the\s+)?(?P<face>left|right|top|bottom|front|back)"
            r"(?:\s+face)?\b",
            re.IGNORECASE,
        )
        ambiguous_pattern = re.compile(
            r"\bload\s+(?:applied\s+)?(?:at|to|on)\s+"
            r"(?:the\s+)?(?:left|right|top|bottom|front|back)(?:\s+face)?"
            r"(?:\s*(?:,|and|or)\s*(?:the\s+)?"
            r"(?:left|right|top|bottom|front|back)(?:\s+face)?)+\b",
            re.IGNORECASE,
        )
        if ambiguous_pattern.search(text):
            raise ValueError("multiple load-application faces are ambiguous")
    else:
        raise ValueError(f"unsupported face action: {action}")
    return tuple(dict.fromkeys(match.group("face").lower() for match in pattern.finditer(text)))


def parse_design_intent(text: str) -> ParsedDesignIntent:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("design intent must be a non-empty string")
    if len(text) > 4000:
        raise ValueError("design intent is longer than the 4000-character parser limit")
    lower = text.lower()

    application: str | None = None
    if "motor" in lower and ("mount" in lower or "bracket" in lower):
        application = "motor mount bracket"
    elif "mount" in lower:
        application = "mount"
    elif "bracket" in lower:
        application = "bracket"

    material_matches = re.findall(r"\b(titanium|aluminum|aluminium|steel)\b", lower)
    materials = set(material_matches)
    if len(materials) > 1:
        raise ValueError("multiple material families are ambiguous")
    material = next(iter(materials), None)
    if material == "aluminium":
        material = "aluminum"

    envelope = _parse_envelope(lower)
    load = _parse_load(lower)
    shaft_diameter = _parse_shaft_diameter(lower)
    shaft_referenced = re.search(r"\bshaft\b", lower) is not None
    shaft_axis = _parse_shaft_axis(lower)
    shaft_center = _parse_shaft_center(lower)
    clearance = _parse_clearance(lower)
    support_faces = _parse_faces(lower, "support")
    load_faces = _parse_faces(lower, "load")
    if len(load_faces) > 1:
        raise ValueError("multiple load-application faces are ambiguous")
    supports = tuple(
        [f"support:{face}" for face in support_faces]
        + [f"load:{face}" for face in load_faces]
    )
    assumptions: list[str] = []
    if shaft_diameter is not None:
        assumptions.append(
            "The stated shaft size is interpreted as diameter; confirm this interpretation."
        )
    elif shaft_referenced:
        missing_shaft_diameter = "shaft diameter"
    else:
        missing_shaft_diameter = None
    if load is not None:
        assumptions.append(
            "Coordinates use x=right, y=forward, z=up; the load is a force vector in newtons."
        )
    missing: list[str] = []
    if envelope is None:
        missing.append("three bounding-envelope dimensions (for example 120 x 80 x 40 mm)")
    if material is None:
        missing.append("material family")
    else:
        missing.append("material grade and verified temperature-dependent allowables")
    if load is None:
        missing.append("a single supported directional load (for example downward 2,000 N)")
    if shaft_diameter is not None:
        if shaft_axis is None:
            missing.append("shaft-axis direction")
        if shaft_center is None:
            missing.append("shaft-center coordinates relative to the envelope")
        if clearance is None:
            missing.append("minimum radial clearance allowance")
    elif missing_shaft_diameter is not None:
        missing.append(missing_shaft_diameter)
    if not support_faces:
        missing.append("explicit support/fixture face")
    if not load_faces:
        missing.append("explicit load-application face")

    ready_for_geometry = (
        envelope is not None
        and (
            not shaft_referenced
            or (
                shaft_diameter is not None
                and shaft_axis is not None
                and shaft_center is not None
                and clearance is not None
            )
        )
    )
    ready_for_analysis = not missing
    return ParsedDesignIntent(
        application=application,
        material_family=material,
        material_grade=None,
        bounding_envelope_m=envelope,
        load_vector_n=load,
        shaft_diameter_m=shaft_diameter,
        shaft_axis=shaft_axis,
        shaft_center_m=shaft_center,
        clearance_allowance_m=clearance,
        supports=supports,
        ready_for_geometry=ready_for_geometry,
        ready_for_analysis=ready_for_analysis,
        missing_requirements=tuple(missing),
        assumptions=tuple(assumptions),
    )


def intent_to_primitive(intent: ParsedDesignIntent) -> BoxWithThroughBores:
    if not intent.ready_for_geometry or intent.bounding_envelope_m is None:
        raise ValueError(
            "intent lacks sufficient geometry constraints: "
            + ", ".join(intent.missing_requirements)
        )
    envelope = intent.bounding_envelope_m
    bores = ()
    if intent.shaft_diameter_m is not None:
        if (
            intent.shaft_axis is None
            or intent.shaft_center_m is None
            or intent.clearance_allowance_m is None
        ):
            raise ValueError("shaft geometry is incomplete")
        if intent.shaft_axis not in {"x", "y", "z"}:
            raise ValueError(f"unsupported shaft axis: {intent.shaft_axis}")
        bore_axis: Axis = intent.shaft_axis
        axis_index = "xyz".index(bore_axis)
        transverse = [index for index in range(3) if index != axis_index]
        bores = (
            ThroughBore(
                axis=bore_axis,
                center=tuple(intent.shaft_center_m[index] for index in transverse),
                radius=intent.shaft_diameter_m / 2 + intent.clearance_allowance_m,
            ),
        )
    model = BoxWithThroughBores(
        center=(0.0, 0.0, 0.0),
        half_extents=tuple(size / 2 for size in envelope),
        bores=bores,
    )
    model.validate()
    return model


def intent_to_sdf(
    intent: ParsedDesignIntent, resolution: int = 32
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    if (
        not isinstance(resolution, (int, np.integer))
        or isinstance(resolution, (bool, np.bool_))
        or resolution < 3
    ):
        raise ValueError("resolution must be an integer of at least 3")
    resolution = int(resolution)
    model = intent_to_primitive(intent)
    if intent.bounding_envelope_m is None:
        raise ValueError("bounding envelope is required to construct an SDF")
    from acwm import make_grid
    from acwm_cad import primitive_sdf

    extents = intent.bounding_envelope_m
    spacing = tuple(size / (resolution - 1) for size in extents)
    origin = tuple(-size / 2 for size in extents)
    points = make_grid((resolution,) * 3, spacing, origin)
    return primitive_sdf(points, model), spacing, origin


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse explicit design constraints without calling an LLM"
    )
    parser.add_argument("--text", required=True, help="Natural-language design request")
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="per-axis resolution for a ready, axis-aligned design envelope",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="export a fully specified supported primitive as STEP or IGES",
    )
    arguments = parser.parse_args(argv)
    if arguments.resolution < 3:
        raise ValueError("resolution must be at least 3")
    result = parse_design_intent(arguments.text)
    payload: dict[str, object] = asdict(result)
    if result.ready_for_geometry:
        phi, spacing, origin = intent_to_sdf(result, arguments.resolution)
        payload["sdf_grid"] = {
            "shape": list(phi.shape),
            "spacing_m": spacing,
            "origin_m": origin,
            "interior_samples": int((phi <= 0).sum()),
        }
    elif arguments.export is not None:
        raise ValueError(
            "cannot export: geometry requirements are missing: "
            + ", ".join(result.missing_requirements)
        )
    if arguments.export is not None:
        from acwm_cad import export_step_iges

        model = intent_to_primitive(result)
        exported = export_step_iges(model, arguments.export)
        payload["cad_export"] = {
            "path": str(exported),
            "format": exported.suffix.lower().lstrip("."),
            "validated_geometry": "axis-aligned box with at most one straight through-bore",
            "analysis_ready": result.ready_for_analysis,
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
