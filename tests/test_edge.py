from __future__ import annotations

import numpy as np
import pytest

from acwm_edge import MacroContext, MacroMicroGovernor, TinyEdgeSurrogate


def test_edge_surrogate_blocks_untrained_inference() -> None:
    torch = pytest.importorskip("torch")
    surrogate = TinyEdgeSurrogate()
    coords = torch.randn(10, 3)
    with pytest.raises(RuntimeError, match="untrained"):
        surrogate(coords, MacroContext(0.05, 1.0))


@pytest.mark.parametrize("shape", ((10, 2), (10, 4), (10,), (0, 3)))
def test_edge_surrogate_validates_coord_shape(shape: tuple[int, ...]) -> None:
    torch = pytest.importorskip("torch")
    surrogate = TinyEdgeSurrogate()
    surrogate.mark_trained()
    with pytest.raises(ValueError, match="Expected coords shape"):
        surrogate(torch.randn(shape), MacroContext(0.05, 1.0))


def test_edge_surrogate_functions_when_trained() -> None:
    torch = pytest.importorskip("torch")
    surrogate = TinyEdgeSurrogate()
    surrogate.mark_trained()
    output = surrogate(torch.randn(5, 3), MacroContext(0.02, 1.2, 0.8))
    assert output.shape == (5, 2)


@pytest.mark.parametrize(
    ("stress", "expected", "reason"),
    (
        (np.array([10.0, 45.0, 80.0]), True, ""),
        (np.array([50.0, 120.0, 90.0]), False, "exceeds allowable"),
        (np.array([]), False, "empty or invalid"),
        (np.array([10.0, np.nan]), False, "non-finite"),
        (np.array([10.0, np.inf]), False, "non-finite"),
        (np.array([-1.0]), False, "negative"),
        (np.array([1.0 + 2.0j]), False, "numeric"),
    ),
)
def test_macro_governor_validates_stress(
    stress: np.ndarray, expected: bool, reason: str
) -> None:
    accepted, message = MacroMicroGovernor(100.0).evaluate_and_filter(stress, True)
    assert accepted is expected
    if reason:
        assert reason in message


def test_macro_governor_rejects_topology_violation() -> None:
    accepted, reason = MacroMicroGovernor(100.0).evaluate_and_filter(
        np.array([10.0]), False
    )
    assert not accepted
    assert "topological" in reason
