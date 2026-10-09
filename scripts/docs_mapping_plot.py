"""Figure for docs/getting_started/labels/mapping.md: labels over a feature trace.

Loads a trial of the Moll2025 example session, reads its ``labels.tsv`` and
``mapping.txt`` through the public helpers, and shades each labelled interval
in its mapping colour.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import ethograph as eto
from ethograph.datasets import dataset_dir
from ethograph.io.session_layout import labels_path
from ethograph.labels.intervals import load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.tsv_store import load_labels_tsv

session = dataset_dir("moll2025")
dt = eto.open(session / "Trial_data.nc")
trial = 41
ds = dt.trial(trial)

mappings = load_label_mapping(session / ".ethograph" / "mapping.txt")
labels = load_labels_tsv(labels_path(session))
labels = labels[labels["trial"] == trial]

speed = ds["speed"].sel(keypoint="beakTip", individual="Crow1")
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(speed["time"], speed, color="black", lw=0.8)
plot_label_segments(ax, labels, mappings, individual="Crow1", alpha=0.4)

used = sorted(labels["labels"].unique())
ax.legend(
    handles=[mpatches.Patch(color=mappings[i]["color"], label=mappings[i]["name"]) for i in used],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    fontsize=7,
    ncol=7,
    frameon=False,
)
ax.set_xlabel("time (s)")
ax.set_ylabel("beakTip speed")
ax.set_title(f"trial {trial}")
fig.tight_layout()
fig.savefig("docs/source/_static/media/mapping_plot_labels.png", dpi=150)
print(used, len(labels))
