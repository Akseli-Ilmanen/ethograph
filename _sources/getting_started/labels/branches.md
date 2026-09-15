(target-label-branches)=
# Label branches

```{warning}
Don't open a new branch unless you need one. Segmentation models treat each
branch as a separate classification problem, so use a second branch only when
labels genuinely overlap in time (e.g. transient events vs. longer states).
```

Within a single branch, each timepoint can only belong to **one label** ---
overlapping labels in the same branch are trimmed/split automatically.
Branches let you keep **independent, overlapping tiers** (e.g. transient
`song`/`peck`/`jump` events vs. longer `active`/`resting`/`sleep` states)
that never trim each other.

Analogous to **git branches**: only one branch is active (editable) at a
time, and changes you make in one branch can never change labels in another.

```{tip}
**A new question is a new branch — and the old branch can help answer it.**
If you trained a segmentation model on one question and your labels are
curated across every session, start the finer question (a moment inside a
state, an event that only follows another) on a new branch and feed the old
branch's labels to the new model as inputs:
{ref}`features.label_inputs <segment-config-label-inputs>`. The two must be
different branches — the config refuses the same one, since a model handed
its own targets learns to copy them.
```

```{raw} html
<video autoplay loop muted playsinline style="width:100%">
  <source src="../../_static/media/branch.mp4" type="video/mp4">
</video>
```

---

## Fixed branch positions

You can have up to **3 branches**, shown simultaneously, each with a fixed draw position:

| Branch | Draws as |
|--------|----------|
| `0` | **Full** --- fills the plot |
| `1` | **Top 1** --- thin top strip |
| `2` | **Top 2** --- thin strip below Top 1 |

If Top 1/Top 2 are shown, Full automatically stops short of them instead of
covering them.

Exactly one branch is **active** (editable) at a time. Only the active
branch's labels can be created, deleted (Ctrl+D) or edited (Ctrl+E) ---
labels on other branches are protected while shown.

Any label you can **see** can be clicked to select it, whatever branch is
active, so playback (V) works on every shown branch — and on a
**prediction** too, clicked on its own predictions panel. Selecting a label
from another branch leaves the active branch alone: it does not change which
label class a new label would be drawn with, and Ctrl+D / Ctrl+E on it are
refused with a message telling you to activate its branch first. A selected
prediction is refused the same way — it is read-only, never editable or
deletable.

- Click a branch's name in the Labels panel to make it active (highlighted).
- Each branch has its own **checkbox** (left of its **x** delete button) to
  show/hide it as an overlay --- independent of which branch is active.
- **Shift+B** swaps the active branch with the previously-active one.
- Drag a label row between branch tables to move it; the mapping file
  updates automatically.
- **+** adds a new branch (max 3); **x** deletes one (must be empty first).
- Imported **predictions** never use a branch strip: each file gets its own
  panel (see {ref}`target-prediction-panels`).

---

## Assigning branches in `mapping.txt`

Add a third column to any line in `mapping.txt` (values `0`-`2`; omitted
defaults to `0`):

```
0 background
1 song 0
2 peck 0
3 jump 0
4 active 1
5 resting 1
6 sleep 1
```

Here `song`/`peck`/`jump` live on branch 0 (Full) and `active`/`resting`/
`sleep` on branch 1 (Top 1) --- a `song` interval and an `active` interval
can overlap freely. See {doc}`mapping` for the full mapping file format.
