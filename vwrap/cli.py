"""Command-line entry point for explicit V-Wrap build and query operations."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from .errors import VWrapError
from .exporter import export
from .feasibility import query_design_space
from .schema import ExportPath, Feature, FeatureKind, FeatureRequest

_KNOWN_REQUEST_KEYS = frozenset({"features"})


def _load_request(path: Path) -> FeatureRequest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VWrapError(f"cannot read request JSON: {error}") from error
    if not isinstance(raw, dict):
        raise VWrapError("request JSON must be an object")
    extra = set(raw) - _KNOWN_REQUEST_KEYS
    if extra:
        raise VWrapError(
            f"unknown key(s) {sorted(extra)}; V-Wrap accepts SI meters only"
        )
    if not isinstance(raw.get("features"), list):
        raise VWrapError("request features must be a list")
    try:
        features = tuple(
            Feature(
                kind=FeatureKind(item["kind"]),
                params=item.get("params", {}),
            )
            for item in raw["features"]
            if isinstance(item, dict)
        )
        if len(features) != len(raw["features"]):
            raise VWrapError("each feature must be an object")
        return FeatureRequest(features=features)
    except (KeyError, TypeError, ValueError) as error:
        raise VWrapError(f"invalid feature request: {error}") from error


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _emit(payload: dict, *, code: int) -> int:
    print(json.dumps(payload, default=_json_default, sort_keys=True))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vwrap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--request", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--path",
        required=True,
        choices=[path.value for path in ExportPath],
    )
    build.add_argument("--resolution", type=int, default=32)

    query = subparsers.add_parser("query")
    query.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        request = _load_request(args.request)
        if args.command == "query":
            return _emit(
                {
                    "status": "ok",
                    "design_space": asdict(query_design_space(request)),
                },
                code=0,
            )
        result = export(
            request,
            args.output,
            path=ExportPath(args.path),
            resolution=args.resolution,
        )
        return _emit({"status": "ok", **asdict(result)}, code=0)
    except VWrapError as error:
        return _emit({"status": "refused", "error": str(error)}, code=2)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
