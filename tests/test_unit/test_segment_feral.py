"""The FERAL export and the way its embeddings come back.

What FERAL reads is a contract nothing on our side enforces: every video's
label list is exactly as long as the video, class ids are contiguous from
0, the splits are by whole trial and the held-out sessions never reach
``train``/``val``. And the config it trains from must be complete, with
the chunk stride derived from the video's own rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import pytest
import xarray as xr
import yaml

import ethograph as eto
from ethograph.io.nwb_alignment import alignment_from_trials_ep
from ethograph.labels.intervals import LABELING_MANUAL
from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv
from ethograph.segment.config import VideoFeaturesConfig, load_config
from ethograph.segment.feral import (
    CONFIG_YAML,
    EMBEDDINGS_DIR,
    EXPORT_YAML,
    LABELS_JSON,
    VIDEOS_TSV,
    export_feral,
    import_feral_predictions,
    load_defaults,
    predictions_file,
    resolve_chunk_step,
)
from ethograph.segment.project import Project
from ethograph.segment.sessions import open_session
from ethograph.video_features.base import check_extractor_name, extractor_module

av = pytest.importorskip("av")

FPS = 10
DURATION = 3.0  # s per trial
N_FRAMES = int(DURATION * FPS)
CAMERA = "cam1"


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for _ in range(n_frames):
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        container.mux(stream.encode(None))


def _make_session(folder: Path, name: str, trials: list[int], *, extra_frames: dict[int, int] | None = None) -> Path:
    """A ``.nc`` session with one individual, a label per trial, a video per trial and an alignment naming it."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, DURATION, 1.0 / FPS)
    datasets, rows, videos = [], [], []
    for trial in trials:
        ds = xr.Dataset(
            {"speed": (("time", "individual"), np.abs(np.sin(t + trial))[:, None])},
            coords={"time": t, "individual": ["A"]},
            attrs={"trial": trial, "fps": FPS},
        )
        datasets.append(ds)
        # class 3 for one second in every trial, class 4 only in odd trials
        rows.append(_row(trial, 3, 0.5, 1.5))
        if trial % 2:
            rows.append(_row(trial, 4, 2.0, 2.5))
        video = f"{name}_trial{trial}_{CAMERA}.mp4"
        _write_video(folder / "videos" / video, N_FRAMES + (extra_frames or {}).get(trial, 0))
        videos.append(video)
    dt = eto.from_datasets(datasets)
    nc_path = folder / f"{name}.nc"
    dt.save(str(nc_path))
    save_labels_tsv(folder / f"{name}_labels.tsv", pd.DataFrame(rows))
    starts = np.arange(len(trials)) * (DURATION + 1.0)
    ep = nap.IntervalSet(start=starts, end=starts + DURATION)
    ep.set_info(trial=np.array(trials), **{f"video_{CAMERA}": np.array(videos)})
    alignment_from_trials_ep(ep, folder / ".ethograph" / "alignment.nwb")
    return nc_path


def _row(trial: int, label: int, on: float, off: float) -> dict:
    return {
        "trial": trial,
        "individual": "A",
        "individual_rec": "",
        "labels": label,
        "onset_s": on,
        "offset_s": off,
        "event_type": "state",
        "confidence": 1.0,
        "labeling_method": LABELING_MANUAL,
        "changepoint_corrected": 0,
        "prediction_source": "",
        "n_samples": N_FRAMES,
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "mapping.txt").write_text("0 background\n3 flap\n4 peck\n5 call\n", encoding="utf-8")
    s1 = _make_session(tmp_path / "sessions" / "s1", "s1", [1, 2, 3, 4])
    s2 = _make_session(tmp_path / "sessions" / "s2", "s2", [1, 2])
    config = {
        "sessions": [
            {"source": str(s1), "video_dir": str(s1.parent / "videos")},
            {"source": str(s2), "video_dir": str(s2.parent / "videos")},
        ],
        "individual": "A",
        "features": {
            "columns": {"feral": {"feral_dims": "0..3"}},
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "video_features": {"extractor": "feral", "camera": CAMERA, "context_s": 2.0, "preset": "lite"},
        "train": {"run_name": "crow", "split": {"train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25}},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_feral_is_a_registered_name_that_cannot_be_built():
    check_extractor_name("feral")
    with pytest.raises(ImportError, match="own environment"):
        extractor_module("feral")


def test_feral_settings_are_refused_for_other_extractors():
    with pytest.raises(ValueError, match="context_s is a FERAL setting"):
        VideoFeaturesConfig(extractor="s3d", context_s=2.0)
    with pytest.raises(ValueError, match="preset is a FERAL setting"):
        VideoFeaturesConfig(extractor="timm", preset="lite")


# ---------------------------------------------------------------------------
# The chunk stride
# ---------------------------------------------------------------------------


def test_chunk_step_spreads_the_chunk_over_the_duration():
    assert resolve_chunk_step(None, 200.0, 64) == 1
    assert resolve_chunk_step(2.0, 200.0, 64) == 6  # 63 gaps * 6 frames = 1.89 s
    assert resolve_chunk_step(0.64, 200.0, 64) == 2
    assert resolve_chunk_step(0.1, 25.0, 64) == 1  # never below every frame


def test_presets_lay_over_the_default_recipe():
    base = load_defaults()
    lite = load_defaults("lite")
    assert lite["backbone"] != base["backbone"] and lite["data"] == base["data"]
    assert load_defaults("max")["data"]["chunk_shift"] == 21
    with pytest.raises(ValueError, match="not a FERAL preset"):
        load_defaults("huge")


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def test_export_writes_feral_layout(project: Path):
    result = Project(project / "config.yaml").export_feral()
    folder = project / "feral"
    assert result.folder == folder and all(p.is_file() for p in result.files)

    labels = json.loads((folder / LABELS_JSON).read_text(encoding="utf-8"))
    assert set(labels) == {"class_names", "is_multilabel", "labels", "splits"}
    assert labels["is_multilabel"] is False
    # contiguous ids from 0, background named as FERAL expects
    assert list(labels["class_names"]) == [str(i) for i in range(len(labels["class_names"]))]
    assert labels["class_names"]["0"] == "other"
    # one label per frame of every video, keys relative to the common prefix
    for video, per_frame in labels["labels"].items():
        assert len(per_frame) == N_FRAMES
        assert (result.prefix / video).is_file()
    assert set(labels["splits"]) == {"train", "val", "test", "inference"}
    assert sorted(labels["splits"]["inference"]) == sorted(labels["labels"])
    roles = labels["splits"]["train"] + labels["splits"]["val"] + labels["splits"]["test"]
    assert sorted(roles) == sorted(labels["labels"])  # every video has exactly one role

    # class 3 covers 0.5–1.5 s of trial 1 of s1: frames 5..15 inclusive-ended
    s1_t1 = next(v for v in labels["labels"] if v.endswith(f"s1_trial1_{CAMERA}.mp4"))
    y = np.asarray(labels["labels"][s1_t1])
    flap = labels["class_names"][str(y[5])]
    assert flap == "flap" and y[4] == 0 and y[15] == y[5] and y[16] == 0

    export = yaml.safe_load((folder / EXPORT_YAML).read_text(encoding="utf-8"))
    assert export["video_fps"] == FPS and export["chunk_step"] == result.chunk_step
    assert export["class_label_ids"][0] == 0 and 5 in export["dropped_label_ids"]  # `call` is never labelled
    videos = pd.read_csv(folder / VIDEOS_TSV, sep="\t")
    assert len(videos) == 6 and set(videos["role"]) <= {"train", "val", "test"}


def test_feral_config_is_complete_and_derived_from_the_rate(project: Path):
    result = Project(project / "config.yaml").export_feral()
    text = (project / "feral" / CONFIG_YAML).read_text(encoding="utf-8")
    assert text.startswith("# Written by ethograph for FERAL")
    cfg = yaml.safe_load(text)
    default = load_defaults("lite")
    assert set(cfg) >= set(default)
    assert cfg["backbone"] == default["backbone"]
    # 2 s at 10 fps over 63 gaps → step 0.32 → 1, but 3 s of video is short; check the arithmetic on the shifts
    assert (
        cfg["data"]["chunk_step"] == result.chunk_step == resolve_chunk_step(2.0, FPS, default["data"]["chunk_length"])
    )
    assert cfg["data"]["chunk_shift"] == default["data"]["chunk_shift"] * result.chunk_step
    assert Path(cfg["data"]["prefix"]) == result.prefix
    assert Path(cfg["data"]["label_json"]) == project / "feral" / LABELS_JSON
    assert cfg["run_name"] == "crow" and Path(cfg["output_dir"]) == project / "feral"


def test_holdout_sessions_never_train_feral(project: Path):
    cfg = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    s2 = cfg["sessions"][1]["source"]
    cfg["train"]["split"]["holdout_sessions"] = [s2]
    (project / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    result = Project(project / "config.yaml").export_feral()
    labels = json.loads((project / "feral" / LABELS_JSON).read_text(encoding="utf-8"))
    s2_videos = {v for v in labels["labels"] if "s2_trial" in v}
    assert s2_videos and s2_videos.isdisjoint(labels["splits"]["train"] + labels["splits"]["val"])
    assert s2_videos <= set(labels["splits"]["test"]) and s2_videos <= set(labels["splits"]["inference"])
    assert result.holdout_sessions == [s2]


def test_video_features_stage_routes_to_the_export(project: Path):
    p = Project(project / "config.yaml")
    written = p.video_features()
    assert (project / "feral" / LABELS_JSON) in written
    with pytest.raises(ValueError, match="never merged"):
        p.video_features(merge=True)


def test_export_refuses_a_second_individual(project: Path):
    cfg = load_config(project / "config.yaml", ["individual=null", "features.individuals=[A, B]"])
    with pytest.raises(ValueError, match="one cropped video per animal"):
        export_feral(cfg)


def test_export_refuses_another_extractor(project: Path):
    cfg = load_config(project / "config.yaml", ["video_features={extractor: s3d}"])
    with pytest.raises(ValueError, match="extractor: feral"):
        export_feral(cfg)


def test_a_video_mostly_outside_its_trial_is_refused(tmp_path: Path, project: Path):
    long = _make_session(tmp_path / "sessions" / "s3", "s3", [1], extra_frames={1: 2 * N_FRAMES})
    cfg = load_config(
        project / "config.yaml",
        [f"sessions=[{{source: '{long.as_posix()}', video_dir: '{(long.parent / 'videos').as_posix()}'}}]"],
    )
    with pytest.raises(ValueError, match="one clip per trial"):
        export_feral(cfg)


# ---------------------------------------------------------------------------
# Predictions coming back
# ---------------------------------------------------------------------------


def _write_predictions(project: Path, result) -> None:
    """``feral infer --output`` for every video folder, predicting the exported labels exactly."""
    labels = json.loads((result.folder / LABELS_JSON).read_text(encoding="utf-8"))
    n_classes = len(labels["class_names"])
    by_json: dict[Path, dict[str, list]] = {}
    for video, per_frame in labels["labels"].items():
        path = predictions_file(result.folder, result.prefix, (result.prefix / video).parent)
        by_json.setdefault(path, {})[Path(video).name] = np.eye(n_classes)[per_frame].tolist()
    for path, preds in by_json.items():
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"preds": preds}), encoding="utf-8")


def test_predictions_come_back_as_the_labels_feral_was_given(project: Path):
    cfg = load_config(project / "config.yaml")
    result = export_feral(cfg)
    _write_predictions(project, result)

    written = import_feral_predictions(cfg)
    assert len(written) == 2  # the two sessions share file-name patterns but never a JSON
    s1 = load_labels_tsv(next(p for p in written if p.name == "s1_predictions.tsv"))
    assert set(s1["labeling_method"]) == {"automated"} and set(s1["prediction_source"]) == {"feral_crow"}
    # FERAL's class ids are compact; what comes back are the project's label ids
    flap = s1[(s1["trial"].astype(str) == "1") & (s1["labels"] == 3)]
    assert len(flap) == 1
    assert flap["onset_s"].iloc[0] == pytest.approx(0.5, abs=1 / FPS)
    assert flap["offset_s"].iloc[0] == pytest.approx(1.5, abs=1 / FPS)
    assert set(s1["labels"]) == set(result.label_ids) - {0}  # a class FERAL never trained on stays background


def test_predictions_from_another_export_are_refused(project: Path):
    cfg = load_config(project / "config.yaml")
    result = export_feral(cfg)
    _write_predictions(project, result)
    path = next((result.folder / "predictions").glob("*.json"))
    preds = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({"preds": {k: v[:-1] for k, v in preds["preds"].items()}}), encoding="utf-8")
    with pytest.raises(ValueError, match="another export"):
        import_feral_predictions(cfg)


def test_import_without_predictions_names_the_commands(project: Path):
    cfg = load_config(project / "config.yaml")
    export_feral(cfg)
    with pytest.raises(FileNotFoundError, match="feral infer"):
        import_feral_predictions(cfg)


# ---------------------------------------------------------------------------
# Embeddings coming back
# ---------------------------------------------------------------------------


def test_embeddings_attach_as_the_feral_variable(project: Path):
    Project(project / "config.yaml").export_feral()
    embeddings = project / "feral" / EMBEDDINGS_DIR
    embeddings.mkdir()
    videos = pd.read_csv(project / "feral" / VIDEOS_TSV, sep="\t")
    for video in videos["video"]:
        np.save(embeddings / f"{Path(video).stem}.npy", np.full((N_FRAMES, 4), 0.5, dtype=np.float32))

    cfg = load_config(project / "config.yaml")
    session = open_session(cfg.sessions[0], cfg)
    ds = session.trial_dataset(1)
    assert ds is not None and "feral" in ds.data_vars
    assert ds["feral"].dims == ("time", "feral_dims") and ds["feral"].shape == (N_FRAMES, 4)
    assert "feral" in session.result.catalog.feature_choices()


def test_missing_embeddings_are_an_error_once_the_column_is_selected(project: Path):
    cfg = load_config(project / "config.yaml")
    with pytest.raises(FileNotFoundError, match="save_embeddings"):
        open_session(cfg.sessions[0], cfg)
    # ... but not while the config does not select the variable (the export itself opens sessions)
    quiet = load_config(project / "config.yaml", ["features.columns={speed: {}}"])
    session = open_session(quiet.sessions[0], quiet)
    assert "feral" not in session.trial_dataset(1).data_vars
