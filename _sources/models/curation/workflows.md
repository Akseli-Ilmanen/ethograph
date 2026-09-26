(target-curation-workflows)=
# Curation workflows

Open **Model ▸ Curation workflows…** in the top bar. A workflow is a list of
custom functions you can chain, so that when you load predictions from a
model's output a few of the steps that follow run for you. The logic is always
the same: **Filter trials** first, then every later step runs over all the
filtered trials.

If you would like a workflow step added, please raise an issue.

## The steps

| Step | What it does |
|------|--------------|
| **Filter trials** | Sets the trials table's column filters, stored by column name so the same workflow runs on any session with that metadata. Every later step runs over exactly the trials the table then shows (see {doc}`../../advanced/metadata`). |
| **Predict onsets** | Runs a trained LightGBM model over those trials, filling classes they do not already carry (see {doc}`../onset_model`). The classes it wrote are passed to a following **Set curation scope** step whose class list is empty. Predictions from the `eto.segment` and `eto.spot` modules are not run here: load them via **File ▸ Import predictions**. |
| **Set curation scope** | Drops label classes into the Curation section's scope area and picks the curation mode. |
| **Label grid view** | Opens the frame grid on the scope, from the chosen cameras, laid out and generated as configured. The workflow waits until the grid is closed. |
| **Video grid** | Opens the clip player on the scope, from the chosen cameras. The workflow waits until the grid is closed. |
| **Segment review** | Starts the label-by-label review over the scope: each label plays, two clicks re-place it, `N` curates. The workflow waits until the review finishes or is stopped; with nothing left to review it is skipped. |
| **Frame-by-frame review** | Starts the boundary review over the scope. The workflow waits until the review finishes or is stopped; with nothing left to review it is skipped. |
| **Curate trials' labels** | Marks every automated label of the chosen classes, in the chosen trials (current, all, filtered or hidden), as curated. An empty class list means whatever the curation scope holds. Curating is not undoable. |
| **Delete trials' labels** | Deletes every label of the chosen classes in the chosen trials, whatever its labeling method. |
| **Purge short labels** | Deletes state-interval labels of the chosen classes shorter than a threshold, in the chosen trials. Point events are never touched. |
| **Stitch labels** | Merges same-class state labels of one individual separated by less than a gap, in the chosen trials — the changepoint correction's own stitch, on its own. Point events are never touched. |
| **Correct changepoints** | Presses the Changepoints tab's manual correction — purge, stitch, snap to changepoints, purge again — with the settings that tab shows, over the current trial or every filtered trial (see {doc}`../../advanced/changepoints/correction`). Those settings are the tab's, not the step's, so set them there first. |
| **Score trials against the model** | Presses the Curation section's **Score now**: each trial's curated labels against what its run predicted, an F1 per trial into the metadata table. Flagging from those scores is the reviewer's decision in the histogram, never a step (see {doc}`difficulty`). |
| **Save labels** | Writes the labels TSV, exactly as `Ctrl+S` does. Nothing reaches disk before this step. |
