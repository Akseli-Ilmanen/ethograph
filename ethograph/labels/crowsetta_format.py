"""Crowsetta format registration for Ethograph.

Registers 'ethograph-seq' as a Crowsetta format: an extended simple-seq with
`individual` and `trial` columns. Import this module to register the format.

Usage:
    import ethograph.labels.crowsetta_format  # registers on import

    scribe = crowsetta.Transcriber(format="ethograph-seq")
    annot = scribe.from_file("labels.tsv").to_annot()
"""

from __future__ import annotations

from pathlib import Path

import crowsetta
import numpy as np
import pandas as pd


@crowsetta.register_format
class EthographSeq(crowsetta.interface.SeqLike):
    """Extended simple-seq format with individual and trial columns."""

    name = "ethograph-seq"
    ext = ".tsv"

    def __init__(
        self,
        onsets_s: np.ndarray,
        offsets_s: np.ndarray,
        labels: np.ndarray,
        individuals: np.ndarray | None = None,
        trials: np.ndarray | None = None,
        annot_path: str | Path = "",
    ):
        self.onsets_s = np.asarray(onsets_s, dtype=np.float64)
        self.offsets_s = np.asarray(offsets_s, dtype=np.float64)
        self.labels = np.asarray(labels)
        self.individuals = (
            np.asarray(individuals) if individuals is not None else np.full(len(self.labels), "ind0", dtype=object)
        )
        self.trials = np.asarray(trials) if trials is not None else np.zeros(len(self.labels), dtype=int)
        self.annot_path = Path(annot_path)

    @classmethod
    def from_file(cls, annot_path: str | Path, **kwargs) -> EthographSeq:
        annot_path = Path(annot_path)
        if annot_path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(annot_path)
        else:
            df = pd.read_csv(annot_path, sep=None, engine="python", encoding="utf-8-sig")
        return cls(
            onsets_s=df["onset_s"].values,
            offsets_s=df["offset_s"].values,
            labels=df["label"].values,
            individuals=df["individual"].values if "individual" in df.columns else None,
            trials=df["trial"].values if "trial" in df.columns else None,
            annot_path=annot_path,
        )

    def to_seq(self) -> crowsetta.Sequence:
        segments = [
            crowsetta.Segment(onset_s=float(o), offset_s=float(f), label=str(lbl))
            for o, f, lbl in zip(self.onsets_s, self.offsets_s, self.labels)
        ]
        return crowsetta.Sequence(
            segments=segments,
            onsets_s=self.onsets_s,
            offsets_s=self.offsets_s,
            labels=self.labels,
        )

    def to_annot(self) -> crowsetta.Annotation:
        return crowsetta.Annotation(
            annot_path=self.annot_path,
            notated_path=None,
            seq=self.to_seq(),
        )

    def to_df(self) -> pd.DataFrame:
        """Return as a DataFrame preserving individual and trial columns."""
        return pd.DataFrame(
            {
                "onset_s": self.onsets_s,
                "offset_s": self.offsets_s,
                "label": self.labels,
                "individual": self.individuals,
                "trial": self.trials,
            }
        )

    def to_file(self, path: str | Path) -> None:
        """Write to TSV."""
        path = Path(path)
        df = self.to_df()
        cols = ["onset_s", "offset_s", "label", "individual", "trial"]
        df = df[[c for c in cols if c in df.columns]]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")

    @classmethod
    def from_intervals_df(
        cls,
        df: pd.DataFrame,
        id_to_name: dict[int, str] | None = None,
        trial=None,
    ) -> EthographSeq:
        """Create from an internal intervals DataFrame (int labels)."""
        if id_to_name is not None:
            labels = df["labels"].map(lambda x: id_to_name.get(int(x), str(x))).values
        else:
            labels = df["labels"].values.astype(str)

        trials = df["trial"].values if "trial" in df.columns else np.full(len(df), trial or 0)

        return cls(
            onsets_s=df["onset_s"].values,
            offsets_s=df["offset_s"].values,
            labels=labels,
            individuals=df["individual"].values,
            trials=trials,
        )
