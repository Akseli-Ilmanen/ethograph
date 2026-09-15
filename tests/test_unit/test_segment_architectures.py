"""Every vendored DLC2Action architecture honours the registry contract.

``model(x: (B, F, T), mask: (B, 1, T)) -> (S, B, C, T)`` with S >= 1, finite
values and zeros on padded frames.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")  # the vendored MotionBERT needs it; the `model` extra declares it

import ethograph as eto  # noqa: E402
import ethograph.segment.models.vendored  # noqa: E402, F401
from ethograph.segment.models import ARCHITECTURES  # noqa: E402
from ethograph.segment.models.vendored import C2F_MIN_FRAMES  # noqa: E402

VENDORED = ["mstcn", "asformer", "c2f_tcn", "c2f_transformer", "edtcn", "mlp", "motionbert", "mp_transformer"]

PARAMS: dict[str, dict[str, Any]] = {"motionbert": {"num_joints": 1}}
"""What an architecture cannot be built without — upstream's ``???`` keys."""
N_FEATURES = 7
N_CLASSES = 3
B = 2
T = 1024
N_PADDED = 100


def _params(name: str) -> dict[str, Any]:
    return dict(PARAMS.get(name, {}))


def _inputs(n_frames: int = T) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N_FEATURES, n_frames)
    mask = torch.ones(B, 1, n_frames)
    mask[1, :, -N_PADDED:] = 0
    return x, mask


def test_registry_contains_vendored_names() -> None:
    assert set(VENDORED) <= set(ARCHITECTURES)


@pytest.mark.parametrize("name", VENDORED)
def test_forward_contract(name: str) -> None:
    model = ARCHITECTURES[name](_params(name), N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    model.train()
    out = model(x, mask)
    assert out.dim() == 4
    assert out.shape[0] >= 1
    assert tuple(out.shape[1:]) == (B, N_CLASSES, T)
    assert torch.isfinite(out).all()
    assert (out[:, 1, :, -N_PADDED:] == 0).all()
    assert (out[:, :, :, : T - N_PADDED] != 0).any()
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", VENDORED)
def test_eval_is_deterministic(name: str) -> None:
    model = ARCHITECTURES[name](_params(name), N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        first = model(x, mask)
        second = model(x, mask)
    assert torch.equal(first, second)


def test_asformer_batch_matches_single_sample() -> None:
    model = ARCHITECTURES["asformer"]({}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs()
    with torch.no_grad():
        batched = model(x, mask)
        singles = [model(x[i : i + 1], mask[i : i + 1]) for i in range(B)]
    assert torch.allclose(batched, torch.cat(singles, dim=1), atol=1e-6)


def test_params_override_defaults() -> None:
    model = ARCHITECTURES["mstcn"]({"num_R": 1}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert model(x, mask).shape[0] == 2


@pytest.mark.parametrize("name", ["c2f_tcn", "c2f_transformer"])
def test_c2f_minimum_frames(name: str) -> None:
    model = ARCHITECTURES[name](_params(name), N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs(C2F_MIN_FRAMES)
    with torch.no_grad():
        assert tuple(model(x, mask).shape[1:]) == (B, N_CLASSES, C2F_MIN_FRAMES)
        x, mask = _inputs(C2F_MIN_FRAMES - 64)
        with pytest.raises(RuntimeError):
            model(x, mask)


#: Registry name -> the upstream config file the builder reads its defaults from.
CONFIG_STEMS = {
    "mstcn": "ms_tcn3",
    "asformer": "asformer",
    "c2f_tcn": "c2f_tcn",
    "c2f_transformer": "c2f_transformer",
    "edtcn": "edtcn",
    "mlp": "mlp",
    "motionbert": "motionbert",
    "mp_transformer": "transformer",
}


@pytest.mark.parametrize("name,stem", sorted(CONFIG_STEMS.items()))
def test_defaults_come_from_the_vendored_upstream_yaml(name: str, stem: str) -> None:
    """No hyperparameter default is written in ``vendored.py``.

    They live in the vendored copies of upstream's own ``config/model``, so a
    vendor refresh moves code and defaults together and the two cannot drift.
    """
    import yaml

    from ethograph.segment.models.vendored import CONFIG_DIR, DATASET_FEATURES, upstream_defaults

    raw = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
    resolved = upstream_defaults(stem, N_FEATURES)
    assert set(resolved) == set(raw), "a key was invented or dropped on the way out of the YAML"
    for key, value in raw.items():
        if value == DATASET_FEATURES:
            assert resolved[key] == {"features": (N_FEATURES,)}
        else:
            assert resolved[key] == value, f"{stem}.{key} was overridden instead of passed through"
    assert DATASET_FEATURES not in resolved.values()


def test_upstream_defaults_refuses_an_unknown_model() -> None:
    from ethograph.segment.models.vendored import upstream_defaults

    with pytest.raises(FileNotFoundError, match="No vendored DLC2Action config"):
        upstream_defaults("not_a_model", N_FEATURES)


def test_every_dlc2action_architecture_maps_to_an_upstream_config() -> None:
    """Every DLC2Action architecture reads its defaults from upstream's YAML.

    The registry also holds the skeleton-graph architectures (``specscalpel``,
    ``lady``), which are not DLC2Action and read their own vendored defaults —
    covered by ``test_segment_skeleton_graph.py`` — and our own ``rnn``, which
    reads ``models/config/rnn.yaml`` (``test_segment_rnn.py``). So the
    DLC2Action stems are a subset of the registry, and everything else is
    exactly those.
    """
    from ethograph.segment.models import DEFAULTS_FILES, available_architectures
    from ethograph.segment.models.skeleton_graph import _DEFAULTS_FILE
    from ethograph.segment.models.vendored import _STEMS

    assert _STEMS == CONFIG_STEMS
    registered = set(available_architectures())
    assert set(_STEMS) <= registered
    assert registered - set(_STEMS) == set(_DEFAULTS_FILE) | set(DEFAULTS_FILES)


@pytest.mark.parametrize("name", sorted(CONFIG_STEMS))
def test_tunable_params_are_exactly_what_build_model_accepts(name: str) -> None:
    """What a search space may name, per architecture.

    The architectures share almost no hyperparameter names, so a benchmark
    over several of them needs this to build one space per architecture.
    """
    from ethograph.segment.models import build_model

    tunable = eto.segment.tunable_params(name)
    assert tunable, name
    assert not any(isinstance(v, str) and v.startswith("dataset_") for v in tunable.values()), (
        "a dataset-derived key is not a hyperparameter the user sets"
    )
    for key, default in tunable.items():
        build_model(name, {**_params(name), key: default}, N_FEATURES, N_CLASSES)


def test_a_param_the_architecture_does_not_take_is_refused_before_training() -> None:
    """Left to the constructor this is a bare ``TypeError``, raised after the
    dataset is materialised and the run directory created — which in a search
    kills the study on some later trial."""
    from ethograph.segment.models import build_model

    with pytest.raises(ValueError, match="not hyperparameters of this architecture"):
        build_model("mlp", {"num_f_maps": 64}, N_FEATURES, N_CLASSES)
    # the message names what it does take
    with pytest.raises(ValueError, match="f_maps_list"):
        build_model("mlp", {"num_f_maps": 64}, N_FEATURES, N_CLASSES)
    # a keyword the builder supplies itself is still settable
    build_model("mstcn", {"exclusive": False}, N_FEATURES, N_CLASSES)


def test_params_override_the_upstream_defaults() -> None:
    from ethograph.segment.models import build_model

    model = build_model("mlp", {"f_maps_list": [8]}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert tuple(model(x, mask).shape) == (1, B, N_CLASSES, T)
    assert [layer.out_channels for layer in model.inner.feature_extractor.layers] == [8, N_CLASSES]


@pytest.mark.parametrize("name", ["c2f_tcn", "c2f_transformer"])
def test_c2f_puts_its_full_resolution_stage_last(name: str) -> None:
    """The contract is that ``logits[-1]`` is the prediction to read.

    C2F emits ``[full-res, T/2, T/4, T/8]`` — finest first — so the adapter
    flips it. Read the wrong end and inference silently gets an 8x-coarse map
    that cannot place a boundary: it is smooth to the point of never changing
    its argmax. Guard on that smoothness, which holds at initialisation.
    """
    from ethograph.segment.models import build_model

    model = build_model(name, {}, N_FEATURES, N_CLASSES).eval()
    x, mask = _inputs(C2F_MIN_FRAMES * 4)
    with torch.no_grad():
        logits = model(x, mask)
    step = [(s[0, :, 1:] - s[0, :, :-1]).abs().mean().item() for s in logits]
    assert step[-1] == max(step), f"stage -1 is not the finest: per-stage mean|Δt| = {step}"
    assert step[-1] > 10 * step[0], f"stages are not ordered coarse -> fine: {step}"


@pytest.mark.parametrize("name", ["motionbert", "mp_transformer"])
def test_windowed_model_returns_its_windows_in_order(name: str) -> None:
    """A fixed-window model's trial is folded into windows and back.

    The fold is a reshape and a permute, and the wrong permute still returns
    ``(S, B, C, T)`` and still trains — it just hands every frame another
    window's prediction. So compare against running the windows by hand.
    """
    model = ARCHITECTURES[name](_params(name), N_FEATURES, N_CLASSES).eval()
    window = model.window
    n_frames = 2 * window + window // 2  # deliberately not a whole number of windows
    x, mask = _inputs(n_frames)
    with torch.no_grad():
        whole = model(x, mask)
        by_hand = torch.cat(
            [model(x[..., i : i + window], mask[..., i : i + window]) for i in range(0, n_frames, window)],
            dim=-1,
        )
    assert tuple(whole.shape) == (1, B, N_CLASSES, n_frames)
    assert torch.allclose(whole, by_hand, atol=1e-5)


def test_build_model_dispatches_vendored() -> None:
    from ethograph.segment.models import available_architectures, build_model

    assert set(VENDORED) <= set(available_architectures())
    model = build_model("mlp", {}, N_FEATURES, N_CLASSES)
    x, mask = _inputs()
    assert tuple(model(x, mask).shape) == (1, B, N_CLASSES, T)
