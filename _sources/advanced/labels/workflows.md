(target-curation-workflows)=
# Curation workflows

Reviewing a model's output is the same handful of moves every session: narrow
the trials table to one condition, run the lightgbm model over what is left, drop
the predicted classes into the curation scope, open a grid laid out the way
that behaviour needs, walk the boundaries, save. A **workflow** is that
sequence written down once and replayed with one button.

Open it from **Model ▸ Curation workflows…** in the top bar — a workflow usually starts with a
prediction, so it sits next to *LightGBM: Predict…*.

Workflows are stored as plain YAML in `~/.ethograph/defaults/workflows/{name}.yaml`,
the same global store as the lightgbm models they invoke — so a workflow written
while curating one dataset is there for the next one.

## Recording one

The fastest way to write a workflow is to do the routine once by hand and
capture it as you go:

1. Set the GUI up for the step (filter the trials table, pick the grid layout,
   set the review window — whatever that step is).
2. In the workflow dialog, pick the step kind and press **Add step**. A new
   step is added already filled in from how the GUI is set up right now.
3. Adjust anything in the form on the right; **Capture current GUI settings**
   re-reads the GUI at any time.

Steps run top to bottom; `↑` / `↓` reorder them and **Remove** drops one.
Everything is saved as you type, so there is no Save button — **New**,
**Rename…**, **Copy** and **Delete** manage the workflows themselves.

## The steps

| Step | What it does |
|------|--------------|
| **Filter trials** | Sets the trials table's column filters. Every later step runs over exactly the trials the table then shows — this is the one trial filter (see {doc}`../metadata`). |
| **Predict onsets** | Runs a trained LightGBM model over those trials, filling classes they do not already carry (see {doc}`../../models/onset_model`). |
| **Set curation scope** | Drops label classes into the Curation section's scope area and picks the curation mode. |
| **Label grid view** | Opens the frame grid on the scope, from the chosen cameras, laid out and generated as configured. |
| **Video grid** | Opens the clip player on the scope, from the chosen cameras. |
| **Frame-by-frame review** | Starts the boundary review over the scope. |
| **Curate trials' labels** | Marks every automated label of the chosen classes, in the chosen trials (current, all, filtered or hidden), as curated — **Curate…** in **Tools ▸ Labels: Bulk editing…**. An empty class list means whatever the curation scope holds. The dialog asks first; a recorded step is already a deliberate choice, so it does not. |
| **Delete trials' labels** | Deletes every label of the chosen classes in the chosen trials — every labeling method, not just automated. |
| **Purge short labels** | Deletes state-interval labels of the chosen classes shorter than a threshold, in the chosen trials. Point events are never touched. |
| **Correct offsets** | Pulls back each label's offset across a near-zero gap to the next onset of the same subject, in the chosen trials, so every interval is strictly separated. Not scoped by label class: a subject's whole sequence has to be seen together. |
| **Save labels** | Writes the labels TSV, exactly as `Ctrl+S` does. |

Each step drives the same widgets you would: there is no second way of
filtering trials, predicting or curating — a workflow is a recording of the
GUI.

### What is carried between steps

Exactly one thing: the label classes the last **Predict onsets** step wrote.
Leave a **Set curation scope** step's class list empty and it scopes to
precisely those classes — so a *predict → scope → grid* workflow reviews what
this run produced, never last week's labels of the same kind.

Everything else a step needs is in the step itself.

### Steps that wait for you

**Label grid view**, **Video grid** and **Frame-by-frame review** are the ones
you actually work through, so the workflow hands over and stops there: the
next step starts when you close the grid, or when the review finishes or is
stopped. The rest of the GUI stays live throughout — the workflow is not
modal.

A **Save labels** step at the end means the session's work is on disk before
you stop paying attention.

```{warning}
**Curate trials' labels** is rarely a follow-up to a grid worked through in
*Click = uncurated, rest = curated*. That mode's **Done** already curates every
unclicked automated label the grid is showing — in the video grid the whole
grid, in the label grid the class its **Label** filter is on (all of them when
it is not filtering) — so a step over the same trials and classes would find
nothing left to do. It only reaches further when its trials or label classes
reach beyond what the grid showed.

It is for the flows where nothing swept up: a grid where **Done** only curated
the handful you clicked, a label grid narrowed to one class so the others were
never judged, a frame-by-frame review you stopped partway, or a workflow with
no review surface at all. That last one is a blanket "a human approved these"
over labels no human saw — the one thing the `automated` / `curated` split
exists to keep apart.
```

```{note}
A **Frame-by-frame review** step with nothing to walk (everything in scope is
already curated) is skipped rather than treated as an error — a workflow run a
second time over the same session simply falls through it.
```


## Filters

A **Filter trials** step stores its conditions by *column name*, not by column
position, so the same workflow runs on any session whose metadata uses those
names. Add one with **Add…** (a categorical column offers its values as
tick-boxes, a numeric column an `≥` / `≤` comparison), or press **From trials
table** to take whatever the table has active right now.

A condition the current dataset cannot honour — a column it does not have, a
value that never occurs in it — is reported in the run log and skipped; the
remaining filters still apply. A workflow written for one cohort therefore
degrades to something sensible on another rather than refusing to run.

## Running

**Run workflow** walks the steps, logging each one and what it did (how many
trials the filter left, what the prediction wrote, how many labels were
curated) in the panel at the bottom. **Stop** abandons the run wherever it is;
nothing already done is undone — a workflow is a sequence of ordinary GUI
actions, and `Ctrl+Z` takes back a label edit here exactly as it does anywhere
else. **Curating is not a label edit**: nothing takes it back, so a workflow
ending in a curate step is a decision, not a draft. Nothing reaches disk until
a **Save labels** step (or `Ctrl+S`), so a run you regret can still be
discarded by closing without saving.

A step that genuinely cannot run — a model that is not on this machine, no
trials table loaded — stops the workflow and says why.
