"""Single source of truth for the feature families Vulcan can represent."""

from .schema import FeatureKind

ANALYTIC_SUPPORTED = frozenset(
    {FeatureKind.BOX, FeatureKind.THROUGH_BORE}
)
LEVEL_SET_SUPPORTED = frozenset(
    {
        FeatureKind.BOX,
        FeatureKind.THROUGH_BORE,
        FeatureKind.SPHERE_EXCLUSION,
        FeatureKind.ROUNDED_BOX,
        FeatureKind.CHAMFER,
        FeatureKind.PATTERN,
    }
)

REFUSED_WITH_REASON = {
    FeatureKind.FILLET_EDGE: (
        "single-edge fillets require a general surface-blend kernel; use "
        "ROUNDED_BOX for an all-edge fillet on the level-set path"
    ),
    FeatureKind.THREADED_HOLE: (
        "threads are helical surfaces; represent the hole as a plain "
        "THROUGH_BORE at the major diameter for screening only"
    ),
    FeatureKind.RIB: (
        "a rib is a second solid body; the analytic family allows exactly "
        "one box, and the level-set path does not guarantee multi-body manifoldness"
    ),
    FeatureKind.FREEFORM: (
        "freeform surfaces are outside both the analytic family and the "
        "supported level-set operators"
    ),
}

FIDELITY_FLOOR = 3.0
