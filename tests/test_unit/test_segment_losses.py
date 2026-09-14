"""The training loss is DLC2Action's ``MS_TCN_Loss``, built from upstream's
own ``config/losses.yaml``.

Nothing on our side reimplements it, so these tests cover the seam: the
defaults come from the vendored YAML, the two keys a config file cannot carry
are filled in, upstream's `dataset_inverse_weights` sentinel is refused rather
than guessed at, an unknown key is refused, and padded frames cost nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ethograph.segment.dataset import PAD_TARGET  # noqa: E402
from ethograph.segment.dlc2action.loss import MS_TCN_Loss  # noqa: E402
from ethograph.segment.losses import (  # noqa: E402
    DATASET_INVERSE_WEIGHTS,
    DEFAULT_TAU,
    build_loss,
    upstream_defaults,
)

N_CLASSES = 3
B, T = 2, 64


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.zeros(4, B, N_CLASSES, T)
    target = torch.zeros(B, T, dtype=torch.long)
    target[:, 20:40] = 1
    return logits, target


def test_defaults_come_from_the_vendored_losses_yaml() -> None:
    """No loss default is written on our side."""
    import yaml

    from ethograph.segment.losses import LOSS_CONFIG, LOSS_KEY

    raw = yaml.safe_load(LOSS_CONFIG.read_text(encoding="utf-8"))[LOSS_KEY]
    assert upstream_defaults() == raw


def test_build_loss_is_upstreams_loss() -> None:
    criterion, settings = build_loss({}, N_CLASSES)
    assert isinstance(criterion, MS_TCN_Loss)
    # the two things the YAML cannot carry
    assert settings["exclusive"] is True
    assert settings["weights"] is None
    # everything else is upstream's, untouched
    for key in ("focal", "gamma", "alpha"):
        assert settings[key] == upstream_defaults()[key]


def test_overrides_beat_the_upstream_defaults() -> None:
    _criterion, settings = build_loss({"focal": False, "gamma": 5}, N_CLASSES)
    assert settings["focal"] is False
    assert settings["gamma"] == 5
    assert settings["alpha"] == upstream_defaults()["alpha"]


def test_an_unknown_loss_key_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        build_loss({"smoothing": 0.15}, N_CLASSES)


def test_the_dataset_weights_sentinel_is_refused_not_guessed() -> None:
    """Resolving it needs DLC2Action's dataset layer, which is not vendored."""
    with pytest.raises(ValueError, match="not vendored"):
        build_loss({"weights": DATASET_INVERSE_WEIGHTS}, N_CLASSES)


def test_explicit_weights_are_passed_through() -> None:
    _criterion, settings = build_loss({"weights": [1.0, 2.0, 3.0]}, N_CLASSES)
    assert settings["weights"] == [1.0, 2.0, 3.0]


def test_padded_frames_cost_nothing() -> None:
    """CE ignores them via ``ignore_index``; the adapter zeroes their logits,
    so a constant log-softmax makes their consistency term exactly zero."""
    criterion, _ = build_loss({}, N_CLASSES)
    logits, target = _batch()

    full = criterion(logits, target)
    padded_target = target.clone()
    padded_target[1, -8:] = PAD_TARGET  # logits there are already zero
    padded = criterion(logits, padded_target)
    assert torch.isfinite(padded)
    assert padded <= full + 1e-6


def test_loss_accepts_the_registry_contract_shape() -> None:
    criterion, _ = build_loss({}, N_CLASSES)
    logits, target = _batch()
    stacked = criterion(logits, target)
    single = criterion(logits[-1], target)
    assert torch.isfinite(stacked) and torch.isfinite(single)


class TestTau:
    """``train.loss.tau`` — the truncation threshold of the consistency term.

    Upstream writes it into the arithmetic (``clamp(..., max=16)``), so the
    contract is: exposing it changes nothing at the default, and moving it
    moves only that term.
    """

    @staticmethod
    def _logits(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch with real boundaries, so the consistency term is not zero."""
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(4, B, N_CLASSES, T, generator=generator) * 4
        target = torch.zeros(B, T, dtype=torch.long)
        target[:, 20:40] = 1
        return logits, target

    def test_the_default_is_upstreams_loss_to_the_last_bit(self) -> None:
        criterion, settings = build_loss({}, N_CLASSES)
        ours = {"tau", "candidate_gate"}
        upstream = MS_TCN_Loss(num_classes=N_CLASSES, **{k: v for k, v in settings.items() if k not in ours})
        logits, target = self._logits()
        assert settings["tau"] == DEFAULT_TAU
        assert torch.equal(criterion(logits, target), upstream(logits, target))

    def test_a_larger_tau_truncates_less(self) -> None:
        """τ bounds the penalty on one frame's jump, so raising it cannot lower the term."""
        logits, _ = self._logits()
        default, _ = build_loss({}, N_CLASSES)
        loose, _ = build_loss({"tau": 48.0}, N_CLASSES)
        tight, _ = build_loss({"tau": 0.5}, N_CLASSES)
        p = logits[-1]
        assert tight.consistency_loss(p) < default.consistency_loss(p) < loose.consistency_loss(p)

    def test_tau_moves_only_the_consistency_term(self) -> None:
        logits, target = self._logits()
        off_default, _ = build_loss({"alpha": 0.0}, N_CLASSES)
        off_loose, _ = build_loss({"alpha": 0.0, "tau": 48.0}, N_CLASSES)
        assert torch.equal(off_default(logits, target), off_loose(logits, target))

    def test_a_non_positive_tau_is_refused(self) -> None:
        with pytest.raises(ValueError, match="truncation threshold"):
            build_loss({"tau": 0.0}, N_CLASSES)


class TestCandidateGate:
    """The consistency term skips the two transitions touching a changepoint candidate — and nothing else."""

    @staticmethod
    def _one_jump(c: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits certain of class 0 before frame *c* and of class 1 from it on; the target agrees."""
        logits = torch.zeros(1, B, N_CLASSES, T)
        logits[..., 0, :c] = 8.0
        logits[..., 1, c:] = 8.0
        target = torch.zeros(B, T, dtype=torch.long)
        target[:, c:] = 1
        return logits, target

    @staticmethod
    def _loss(candidate_gate: bool):
        from ethograph.segment.losses import TruncatedMSTCNLoss

        return TruncatedMSTCNLoss(num_classes=N_CLASSES, focal=False, alpha=1.0, candidate_gate=candidate_gate)

    def test_no_candidates_is_upstreams_loss(self) -> None:
        logits, target = self._one_jump(30)
        none = torch.zeros(B, T, dtype=torch.bool)
        gated = self._loss(True)(logits, target, none)
        plain = MS_TCN_Loss(num_classes=N_CLASSES, focal=False, alpha=1.0)(logits, target)
        assert torch.isclose(gated, plain)

    def test_a_candidate_exempts_the_jump_into_and_out_of_it(self) -> None:
        """A jump at frame ``c`` is free with a candidate at ``c`` or ``c - 1``, and costs with one further away."""
        c = 30
        logits, target = self._one_jump(c)
        ce_only = self._loss(True)._ce_loss(logits[0], target)

        def cost(frame: int) -> float:
            candidates = torch.zeros(B, T, dtype=torch.bool)
            candidates[:, frame] = True
            return float(self._loss(True)(logits, target, candidates) - ce_only)

        assert cost(c) == pytest.approx(0.0, abs=1e-6)
        assert cost(c - 1) == pytest.approx(0.0, abs=1e-6)
        assert cost(c + 1) > 1e-3
        assert cost(c - 2) > 1e-3
        # ...and that cost is exactly upstream's consistency term, averaged over one fewer pair of transitions
        ungated = float(self._loss(False)(logits, target) - ce_only)
        assert cost(c + 1) == pytest.approx(ungated * (T - 1) / (T - 3), rel=1e-4)

    def test_the_gate_is_off_by_default_and_ignores_candidates_then(self) -> None:
        c = 30
        logits, target = self._one_jump(c)
        candidates = torch.zeros(B, T, dtype=torch.bool)
        candidates[:, c] = True
        assert torch.isclose(self._loss(False)(logits, target, candidates), self._loss(False)(logits, target))

    def test_build_loss_resolves_the_gate_from_the_layout(self) -> None:
        from ethograph.segment.losses import CANDIDATE_GATE_KEY

        assert build_loss({}, N_CLASSES, has_candidates=True)[1][CANDIDATE_GATE_KEY] is True
        assert build_loss({}, N_CLASSES, has_candidates=False)[1][CANDIDATE_GATE_KEY] is False
        assert build_loss({}, N_CLASSES)[1][CANDIDATE_GATE_KEY] is False
        assert build_loss({CANDIDATE_GATE_KEY: False}, N_CLASSES, has_candidates=True)[1][CANDIDATE_GATE_KEY] is False
        with pytest.raises(ValueError, match="no changepoint candidate column"):
            build_loss({CANDIDATE_GATE_KEY: True}, N_CLASSES, has_candidates=False)
