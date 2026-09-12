"""Export a trained segment run's split to FERAL's input: one video per trial + a label JSON.

FERAL (repos/feral) learns per-frame classes from whole video files. Each sample
of the materialised dataset is one trial of one individual, and each trial has
its own camera file, so a FERAL "video" is a trial's camera file as recorded —
no cutting, no re-encoding. Labels are the materialised ``groundTruth`` (the
exact rasterisation the segment run trained on), class indices unchanged, class
0 renamed ``"other"`` so FERAL treats it as background. The split is the run's
own ``splits/*.bundle``, less the trials that have no video (``videos_missing.tsv``;
score_feral.py re-scores the segment run on the same subset); test trials are also listed as FERAL's ``inference``
split, which is the only output FERAL writes with per-frame probabilities.

Usage (ethograph env):
    python export_feral.py RUN_DIR OUT_DIR --video-root C:/Users/aksel/Documents/VidData --camera cam-1
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import av
import pandas as pd
import yaml

from ethograph.segment.config import load_config
from ethograph.segment.materialise import load_sample, read_classes, read_index
from ethograph.segment.sessions import Session, open_session

ROLES = ("train", "val", "test")
CHUNK_LENGTH = 64  # FERAL's predict_per_item; one query per frame of a chunk


def session_video_dir(source: Path, video_root: Path) -> Path:
    """``.../sub-03_id-Freddy/ses-000_date-20250526_01/behav/x.nc`` → ``{video_root}/20250526_01_Freddy``."""
    date = re.search(r"date-(\w+)", str(source))
    subject = re.search(r"id-([A-Za-z]+)", str(source))
    if date is None or subject is None:
        raise ValueError(f"Cannot read date/subject off {source}")
    return video_root / f"{date.group(1)}_{subject.group(1)}"


def frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return int(container.streams.video[0].frames)


def video_rate(path: Path) -> float:
    with av.open(str(path)) as container:
        return float(container.streams.video[0].average_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A finished segment run (config.yaml + splits/)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--camera", default="cam-1")
    parser.add_argument("--context-s", type=float, default=2.0, help="Seconds one FERAL chunk should span")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    config = load_config(run_dir / "config.yaml", [])
    data_dir = config.data_dir
    classes = read_classes(data_dir)
    index = read_index(data_dir).set_index("key")
    roles = {
        role: (run_dir / "splits" / f"{role}.bundle").read_text(encoding="utf-8").split() for role in ROLES
    }

    sessions: dict[str, Session] = {}
    for spec in config.sessions:
        spec = replace(spec, video_dir=session_video_dir(spec.source, args.video_root))
        session = open_session(spec, None, expand_changepoints=False)
        sessions[session.id] = session

    labels: dict[str, list[int]] = {}
    splits: dict[str, list[str]] = {role: [] for role in ROLES}
    rows = []
    missing = []
    rates: set[float] = set()
    for role, keys in roles.items():
        for key in keys:
            sample = index.loc[key]
            session = sessions[sample["session_id"]]
            trial = int(sample["trial"])
            offset = session.result.nwb_alignment.stream_offset_for_trial(trial, "video", device=args.camera)
            if offset != 0:
                raise ValueError(f"{key}: video offset {offset} s — a trial clip would have to be cut first")
            video = session.media_path(trial, "video", args.camera)
            if video is None or not video.is_file():
                missing.append({"key": key, "role": role, "session_id": sample["session_id"], "trial": trial})
                continue
            _, y = load_sample(data_dir, key, classes)
            n_video = frame_count(video)
            if n_video != len(y):
                raise ValueError(f"{key}: {video.name} has {n_video} frames, labels {len(y)}")
            rates.add(video_rate(video))
            name = video.relative_to(args.video_root).as_posix()
            labels[name] = [int(v) for v in y]
            splits[role].append(name)
            rows.append({"video": name, "key": key, "role": role, "session_id": sample["session_id"], "trial": trial})

    if len(rates) != 1:
        raise ValueError(f"Videos disagree on frame rate: {sorted(rates)}")
    fps = rates.pop()
    chunk_step = max(1, round(args.context_s * fps / (CHUNK_LENGTH - 1)))
    span = (CHUNK_LENGTH - 1) * chunk_step + 1

    out: Path = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    class_names = {str(i): ("other" if i == 0 else name) for i, name in enumerate(classes.names)}
    label_json = {
        "class_names": class_names,
        "is_multilabel": False,
        "labels": labels,
        "splits": {**splits, "inference": list(splits["test"])},
    }
    (out / "labels.json").write_text(json.dumps(label_json), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "videos.tsv", sep="\t", index=False)
    pd.DataFrame(missing, columns=["key", "role", "session_id", "trial"]).to_csv(
        out / "videos_missing.tsv", sep="\t", index=False
    )
    if missing:
        by_role = pd.DataFrame(missing)["role"].value_counts().to_dict()
        print(f"{len(missing)} trials have no {args.camera} video and are left out: {by_role} (videos_missing.tsv)")
    overrides = {
        "run_name": f"feral_lite_{run_dir.name}",
        "data": {
            "prefix": str(args.video_root.resolve()),
            "label_json": str(out / "labels.json"),
            "chunk_step": chunk_step,
            "chunk_shift": span // 2,  # lite's 50 % overlap, in frames of the span
        },
        "model": {"gradient_checkpointing": True},
        "training": {"compile": False},
        "segment_run": str(run_dir),
        "video_fps": fps,
        "context_s": span / fps,
    }
    (out / "feral_overrides.yaml").write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    print(
        f"{sum(len(v) for v in splits.values())} videos ({', '.join(f'{r} {len(splits[r])}' for r in ROLES)}), "
        f"{len(class_names)} classes, {fps:g} fps → chunk_step {chunk_step}, span {span} frames "
        f"({span / fps:.2f} s) → {out}"
    )


if __name__ == "__main__":
    main()
