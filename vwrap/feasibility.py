"""Dynamic design-space query without an implicit export-path preference."""
from __future__ import annotations

from .capability import (
    ANALYTIC_SUPPORTED,
    LEVEL_SET_SUPPORTED,
    REFUSED_WITH_REASON,
)
from .schema import (
    ActionDescriptor,
    CandidateAddition,
    DesignSpace,
    ExportPath,
    FeatureKind,
    FeatureRequest,
    PathEffect,
    RequestState,
)


def _compatible_paths(kinds: tuple[FeatureKind, ...]) -> frozenset[ExportPath]:
    boxes = sum(kind is FeatureKind.BOX for kind in kinds)
    bores = sum(kind is FeatureKind.THROUGH_BORE for kind in kinds)
    paths = set()
    if (
        boxes == 1
        and bores <= 1
        and all(kind in ANALYTIC_SUPPORTED for kind in kinds)
    ):
        paths.add(ExportPath.ANALYTIC)
    if boxes == 1 and all(kind in LEVEL_SET_SUPPORTED for kind in kinds):
        paths.add(ExportPath.LEVEL_SET)
    return frozenset(paths)


def _effect(
    before: frozenset[ExportPath],
    after: frozenset[ExportPath],
) -> PathEffect:
    if before == after:
        return PathEffect.NO_OP if not after else PathEffect.PRESERVES
    if not before and after:
        return PathEffect.INITIALIZES
    if not after:
        return PathEffect.BREAKS
    if after < before:
        return PathEffect.DOWNGRADES
    raise AssertionError("adding a feature must not expand compatible paths")


def _blocking_reasons(
    kinds: tuple[FeatureKind, ...],
    paths: frozenset[ExportPath],
) -> tuple[str, ...]:
    if paths:
        return ()
    reasons = [
        f"{kind.value} is globally refused: {REFUSED_WITH_REASON[kind]}"
        for kind in kinds
        if kind in REFUSED_WITH_REASON
    ]
    boxes = sum(kind is FeatureKind.BOX for kind in kinds)
    if boxes != 1:
        reasons.append(
            f"{boxes} base boxes present; every export path requires exactly 1"
        )
    if not reasons:
        reasons.append("no export path is compatible with the current request")
    return tuple(reasons)


def query_design_space(request: FeatureRequest) -> DesignSpace:
    kinds = request.kinds()
    paths = _compatible_paths(kinds)
    if not request.features:
        state = RequestState.EMPTY
    elif not paths:
        state = RequestState.UNSATISFIABLE
    else:
        state = RequestState.VALID

    candidates = tuple(
        CandidateAddition(
            kind=kind,
            effect=_effect(paths, _compatible_paths(kinds + (kind,))),
            compatible_paths_after=_compatible_paths(kinds + (kind,)),
        )
        for kind in FeatureKind
    )
    actions = []
    if state is RequestState.EMPTY:
        actions.append(
            ActionDescriptor(
                "add",
                FeatureKind.BOX,
                PathEffect.INITIALIZES,
                "a base box is required to initialize either export path",
            )
        )
    for candidate in candidates:
        if candidate.effect in (PathEffect.BREAKS, PathEffect.NO_OP):
            continue
        rationale = {
            PathEffect.INITIALIZES: "opens the request; every path requires a base box",
            PathEffect.PRESERVES: "preserves all currently compatible paths",
            PathEffect.DOWNGRADES: "narrows compatible paths to "
            + ", ".join(sorted(path.value for path in candidate.compatible_paths_after)),
        }[candidate.effect]
        actions.append(
            ActionDescriptor("add", candidate.kind, candidate.effect, rationale)
        )
    if state is RequestState.UNSATISFIABLE:
        for reason in _blocking_reasons(kinds, paths):
            actions.append(ActionDescriptor("remove", FeatureKind.BOX, None, reason))

    return DesignSpace(
        state=state,
        compatible_paths=paths,
        next_actions=tuple(actions),
        candidate_additions=candidates,
        blocking_reasons=_blocking_reasons(kinds, paths),
        refused_categories=tuple(
            (kind.value, reason) for kind, reason in REFUSED_WITH_REASON.items()
        ),
    )
