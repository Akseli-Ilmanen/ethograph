"""Score FERAL's test predictions with the segment pipeline's own metrics.

Reads FERAL's inference JSON (per-frame ``(T, C)`` probabilities for the test
videos), takes the argmax exactly as a segment run decodes, and evaluates it
against the materialised ground truth with ``evaluate_dense`` — the code that
scored the segment run the export came from. Writes:

* ``{project}/runs/feral_lite_{segment run}/`` — a run folder ``compare_runs``
  reads (config.yaml, classes.yaml, test_metrics.yaml, test_eval.npz, eval.pdf);
* one prediction set per session under the session's ``labels/`` folder, so the
  test trials open in the GUI beside the curated labels.

``postprocessed`` uses the segment run's post-processing *without* its
``label_thresholds``: those were tuned on the segment model's probabilities.

Usage (ethograph env):
    python score_feral.py OUT_DIR [INFERENCE_JSON]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.labels.intervals import LABELING_AUTOMATED, NO_RECIPIENT
from ethograph.labels.ml import dense_to_intervals
from ethograph.labels.onset_curves import write_provenance
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import load_config
from ethograph.segment.inference import prediction_run_dir
from ethograph.segment.materialise import load_sample, read_classes, read_index
from ethograph.segment.metrics import save_eval_arrays, scalar_metrics
from ethograph.segment.plotting import write_eval_pdf
from ethograph.segment.postprocess import postprocess_dense, postprocess_intervals
from ethograph.segment.train import EVAL_ARRAYS_FILE, TEST_METRICS_FILE, compare_runs, evaluate_dense


def latest_inference_json(out: Path) -> Path:
    found = sorted((out / "answers").glob("_inference_*.json"))
    if not found:
        raise FileNotFoundError(f"No FERAL inference JSON under {out / 'answers'} — has run_feral.py finished?")
    return found[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("inference_json", type=Path, nargs="?")
    args = parser.parse_args()

    out: Path = args.out_dir.resolve()
    overrides = yaml.safe_load((out / "feral_overrides.yaml").read_text(encoding="utf-8"))
    feral_cfg = yaml.safe_load((out / "feral_config.yaml").read_text(encoding="utf-8"))
    segment_run = Path(overrides["segment_run"])
    config = load_config(segment_run / "config.yaml", [])
    data_dir = config.data_dir
    classes = read_classes(data_dir)
    index = read_index(data_dir).set_index("key")
    fs = float(index["fs"].iloc[0])
    thresholds = list(config.train.f1_thresholds)
    pcfg = replace(config.infer.postprocess, label_thresholds={})

    inference_json = args.inference_json or latest_inference_json(out)
    preds = json.loads(inference_json.read_text(encoding="utf-8"))["preds"]
    videos = pd.read_csv(out / "videos.tsv", sep="\t")
    test = videos[videos["role"] == "test"]

    gt: dict[str, np.ndarray] = {}
    pred: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    for row in test.itertuples():
        _, y = load_sample(data_dir, row.key, classes)
        p = np.asarray(preds[row.video], dtype=np.float32)
        if p.shape != (len(y), len(classes.names)):
            raise ValueError(f"{row.key}: FERAL predicted {p.shape}, ground truth is ({len(y)}, {len(classes.names)})")
        gt[row.key], pred[row.key], probs[row.key] = y, p.argmax(axis=1), p

    processed_pred = {key: postprocess_dense(value, fs, classes, pcfg) for key, value in pred.items()}
    raw = evaluate_dense(classes, gt, pred, thresholds, fs)
    processed = evaluate_dense(classes, gt, processed_pred, thresholds, fs)

    run_name = feral_cfg["run_name"]
    run_dir = config.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = yaml.safe_load((segment_run / "config.yaml").read_text(encoding="utf-8"))
    run_config["model"] = {"architecture": f"feral:{feral_cfg['backbone']}", "params": {}}
    run_config["infer"]["postprocess"]["label_thresholds"] = {}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    shutil.copy(segment_run / "classes.yaml", run_dir / "classes.yaml")
    shutil.copy(out / "feral_config.yaml", run_dir / "feral_config.yaml")
    test_metrics = {
        "best_epoch": None,  # FERAL keeps its best val frame-mAP checkpoint; the epoch is in its log
        "train_seconds": None,
        "select_on": "feral val frame_level_map",
        "thresholds": thresholds,
        "segment_run": str(segment_run),
        "inference_json": str(inference_json),
        "postprocess_note": "segment run's infer.postprocess without label_thresholds",
        "raw": {**scalar_metrics(raw), "classwise": raw["classwise"]},
        "postprocessed": {**scalar_metrics(processed), "classwise": processed["classwise"]},
    }
    (run_dir / TEST_METRICS_FILE).write_text(yaml.safe_dump(test_metrics, sort_keys=False), encoding="utf-8")
    save_eval_arrays(run_dir / EVAL_ARRAYS_FILE, raw, processed)
    write_eval_pdf(run_dir / "eval.pdf", raw, processed, classes, thresholds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for session_id, rows in test.groupby("session_id"):
        source = Path(index.loc[rows["key"].iloc[0], "source"])
        records = []
        arrays: dict[str, np.ndarray] = {}
        for row in rows.itertuples():
            individual = str(index.loc[row.key, "individual"])
            p = probs[row.key]
            time = np.arange(len(p)) / fs
            intervals = postprocess_intervals(dense_to_intervals(classes.ids(pred[row.key]), [individual], time_coord=time), pcfg)
            arrays[row.key] = p.astype(np.float16)
            for seg in intervals.itertuples():
                lid = int(seg.labels)
                span = (time >= seg.onset_s) & (time <= seg.offset_s)
                column = classes.label_ids.index(lid)
                records.append(
                    {
                        "trial": row.trial,
                        "individual": individual,
                        "individual_rec": NO_RECIPIENT,
                        "labels": lid,
                        "onset_s": float(seg.onset_s),
                        "offset_s": float(seg.offset_s),
                        "event_type": "state",
                        "confidence": float(p[span, column].mean()),
                        "labeling_method": LABELING_AUTOMATED,
                        "changepoint_corrected": 0,
                        "prediction_source": run_name,
                        "n_samples": len(p),
                    }
                )
        pred_dir = prediction_run_dir(source, run_name, timestamp)
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_labels_tsv(pred_dir / f"{source.stem}_predictions.tsv", pd.DataFrame(records))
        np.savez_compressed(pred_dir / f"{source.stem}_probs.npz", **arrays)
        write_provenance(
            pred_dir,
            model_config=run_dir / "config.yaml",
            inference={"model": "feral", "run": run_name, "session": str(source), "trials": "test split only"},
        )
        print(f"{session_id}: {len(records)} labels over {len(rows)} test trials → {pred_dir}")

    select_on = config.train.select_on
    print(f"FERAL test {select_on}: raw {raw[select_on]:.2f}, post-processed {processed[select_on]:.2f} → {run_dir}")
    print(compare_runs(config.runs_dir).to_string())


if __name__ == "__main__":
    main()
