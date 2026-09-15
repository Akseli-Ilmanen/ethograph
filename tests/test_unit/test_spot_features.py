"""The pose side is the listed features and nothing else: export, scale, fingerprint, and the student's input.

What earns a test: the block must be the same columns on the strided clock,
z-scored on the *training* split and reused at that scale for a session
predicted later; the vendored model must actually read the block (and refuse
to run without one once trained with it); zeroing it must be a real ablation
— different output, same shape; and the old spellings (``graph:``, ``fuse:``)
must be refused by name rather than ignored.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ethograph.spot.config import config_from_dict
from ethograph.spot.e2espot.dataset.frame import load_side_clip
from ethograph.spot.e2espot.train_e2e import E2EModel
from ethograph.spot.features import (
    NAMES_FILE,
    STATS_FILE,
    block_dim,
    export_block,
    export_block_for_inference,
    feature_names,
    write_trial_features,
)
from ethograph.spot.project import Project

FEATURES = {"speed": {"keypoint": ["beakTip", "stickTip"]}, "pellet_stickClosest_dist": {}}


def _config(tmp_path, **extra):
    source = tmp_path / "ses-01.nc"
    source.touch()
    data = {"sessions": [str(source)], "labels": {"classes": [31, 32]}, "root": str(tmp_path), "features": FEATURES}
    data.update(extra)
    return config_from_dict(data, tmp_path)


def _features(config, videos, n=400, fps=200.0, seed=0):
    names = feature_names(config.features)
    config.features_dir.mkdir(parents=True, exist_ok=True)
    (config.features_dir / NAMES_FILE).write_text(json.dumps({"names": names}))
    rng = np.random.default_rng(seed)
    time = np.arange(n) / fps
    for i, video in enumerate(videos):
        x = rng.normal(size=(n, len(names))).astype(np.float32) * (i + 1) + i
        x[5, 0] = np.nan
        write_trial_features(config.features_dir / f"{video}.npz", time, x, {31: 0.5, 32: 1.2})
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    header = "key\tsource\ttrial\tfps\tnum_frames\tnum_events"
    rows = ["\t".join(map(str, (v, config.sessions[0].source, i, fps, n, 2))) for i, v in enumerate(videos)]
    (config.dataset_dir / "index.tsv").write_text("\n".join([header, *rows]) + "\n")
    for split, ids in (("train", videos[:2]), ("val", videos[2:3]), ("test", videos[3:])):
        (config.dataset_dir / f"{split}.json").write_text(
            json.dumps([{"video": v, "fps": fps, "num_frames": n, "events": []} for v in ids])
        )
    return names


class TestConfig:
    def test_listed_features_are_fed_in_unless_kept_for_the_teacher(self, tmp_path):
        assert _config(tmp_path).fusing
        assert not _config(tmp_path, train={"features_as_input": False}).fusing
        assert not _config(tmp_path, features={}).fusing

    def test_the_old_spellings_are_refused_by_name(self, tmp_path):
        with pytest.raises(ValueError, match="graph: is gone"):
            _config(tmp_path, graph={"nodes": ["beakTip"]})
        with pytest.raises(ValueError, match="fuse: is gone"):
            _config(tmp_path, fuse={"enabled": True})
        with pytest.raises(ValueError, match="top level"):
            _config(tmp_path, teacher={"extra_features": {"speed": {}}})
        with pytest.raises(ValueError, match="lists the pose variables directly"):
            _config(tmp_path, features={"columns": {}})

    def test_dropout_is_a_share(self, tmp_path):
        with pytest.raises(ValueError, match="features_dropout"):
            _config(tmp_path, train={"features_dropout": 1.0})

    def test_the_run_is_named_after_the_clip_plus_features(self, tmp_path):
        cfg = _config(tmp_path)
        _features(cfg, ["a", "b", "c", "d"])
        assert Project(cfg).run_name().endswith("_features")
        assert not Project(_config(tmp_path, features={})).run_name().endswith("_features")


class TestBlock:
    def test_block_is_the_columns_on_the_strided_clock_scaled_on_train(self, tmp_path):
        cfg = _config(tmp_path)
        names = _features(cfg, ["a", "b", "c", "d"])
        out = export_block(cfg)
        assert block_dim(cfg) == len(names) == 3
        stride = cfg.clip.resolve(200.0).stride
        with np.load(out / "a.npz") as npz:
            assert npz["features"].shape == (400 // stride, 3)
            assert int(npz["stride"]) == stride
            assert np.isfinite(npz["features"]).all()  # the NaN reads as 0
        with np.load(out / STATS_FILE) as npz:
            assert npz["mean"].shape == (3,)
        assert {p.stem for p in out.glob("*.npz")} == {"a", "b", "c", "d", "stats"}

    def test_a_later_session_is_put_on_the_training_scale(self, tmp_path):
        cfg = _config(tmp_path)
        _features(cfg, ["a", "b", "c", "d"])
        export_block(cfg)
        with np.load(cfg.block_dir / "d.npz") as npz:
            trained_scale = npz["features"].copy()
        stats_before = (cfg.block_dir / STATS_FILE).read_bytes()
        export_block_for_inference(cfg, ["d"])
        with np.load(cfg.block_dir / "d.npz") as npz:
            np.testing.assert_allclose(npz["features"], trained_scale)
        assert (cfg.block_dir / STATS_FILE).read_bytes() == stats_before


@pytest.fixture(scope="module")
def e2e_model():
    """The vendored model on CPU, an untrained tiny backbone, with a 3-wide feature block."""
    # modality 'bw' skips the pretrained download; no shift module, so any clip length goes
    return E2EModel(3, "rny002", "gru", clip_len=8, modality="bw", device="cpu", fuse_dim=3)


class TestVendoredModel:
    def test_reads_the_block_and_zeroing_it_changes_the_answer(self, e2e_model):
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)
        seq = torch.rand(1, 8, 1, 64, 64)
        block = torch.rand(1, 8, 3) * 5
        _, with_features = e2e_model.predict(seq, use_amp=False, fuse=block)
        _, zeroed = e2e_model.predict(seq, use_amp=False, fuse=torch.zeros(1, 8, 3))
        assert with_features.shape == zeroed.shape == (1, 8, 3)
        assert not np.allclose(with_features, zeroed)

    def test_refuses_to_run_without_the_block_it_was_built_for(self, e2e_model):
        torch = pytest.importorskip("torch")
        with pytest.raises(ValueError, match="pose block"):
            e2e_model.predict(torch.rand(1, 8, 1, 64, 64), use_amp=False)

    def test_one_block_serves_every_augmented_view(self, e2e_model):
        torch = pytest.importorskip("torch")
        seq = torch.rand(2, 8, 1, 64, 64)  # a clip and its flip
        _, scores = e2e_model.predict(seq, use_amp=False, fuse=torch.rand(8, 3))
        assert scores.shape == (2, 8, 3)


class TestSideClip:
    def test_zero_hands_back_zeros_and_a_foreign_stride_is_refused(self, tmp_path):
        np.savez(tmp_path / "v.npz", features=np.arange(20, dtype=np.float32).reshape(10, 2), stride=2, fps=200.0)
        clip, mask = load_side_clip(str(tmp_path), "features", {}, "v", base_idx=4, clip_len=4, stride=2)
        np.testing.assert_array_equal(clip[:, 0], [4, 6, 8, 10])  # strided index 2..5
        assert mask.all()
        zeros, mask = load_side_clip(str(tmp_path), "features", {}, "v", 4, 4, 2, zero=True)
        assert not zeros.any() and not mask.any()
        with pytest.raises(ValueError, match="stride"):
            load_side_clip(str(tmp_path), "features", {}, "v", 4, 4, stride=1)
