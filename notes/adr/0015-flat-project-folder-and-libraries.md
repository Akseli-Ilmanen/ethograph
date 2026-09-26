# 0015 — The project folder is flat; skeletons and geometries are libraries resolved like `mapping.txt`

## Context

The project folder grew a `config/` layer holding `skeleton/`, `space/` and the
two pipeline YAMLs, and the docs drew it that way — while the code read only
`config/skeleton/` from a project, read geometries from the home folder alone,
and never looked for a pipeline YAML anywhere: `segment.yaml` / `spot.yaml` are
passed to `Project()` by path, and each pipeline's `root` defaults to the YAML's
own folder, where `runs/`, `data/` or `dataset/` and `runs/compare.tsv` grow.
Two YAMLs in one folder therefore share a `runs/` and overwrite each other's
compare table, whichever folder that is. Reference geometries lived in
`~/.ethograph/defaults/config/space/` only because the Moll et al., 2025
template needed its arena somewhere a fresh install could find it.

## Decision

1. **The project folder is flat.** `mapping.txt`, `skeleton/`, `space/`,
   `workflows/`, `wizard/` sit at the root; there is no `config/` layer, since
   it would hold one thing. The GUI adds `sessions.txt`. The home layout
   migration (`HOME_LAYOUT_MOVES`) moves an install's `defaults/config/*` across
   once and removes the emptied folder; no released version wrote
   `config/skeleton/` into a user's project, so projects need no fallback.

2. **Each pipeline gets a folder of its own**: `segment/segment.yaml`,
   `spot/spot.yaml`, with `root: .` so its data and runs grow beside the file.
   The bundled examples and the starter project have the same shape. The docs
   say this in one place (folder layout) and the quickstarts point there.

3. **`skeleton/` and `space/` are libraries resolved nearest first, the rule
   `mapping.txt` already follows**: the session's `.ethograph/<name>/`, then the
   project's `<name>/`, then the starter project's `~/.ethograph/defaults/<name>/`
   (`skeleton/library.py: library_dirs`, `plots_space.py: geometry_dirs`). A
   name in two libraries resolves to the nearer file. The home folder ships no
   geometry any more: the Moll 2025 template writes its arena
   (`ethograph/assets/space/moll2025.yaml`) into its own session's
   `.ethograph/space/` through the template's `configs`, which now accept a
   subfolder in the name and a bundled `Path` as the value.

4. **OCTRON (box labelling) is paused and appears in no layout, doc or
   docstring.** Its code and its `octron/` folder stay as they are, and the
   bundled `octron.yaml` moved with the rest into `defaults/octron/`, but
   nothing public describes it until the feature is picked up again.

## Consequences

- The folder-layout page, the mapping page, the installation page, the
  bundled `defaults/README.md` and `project.py`'s docstring draw one tree.
- The Space controls' combo re-scans on session and project change, so a
  geometry dropped into a session shows up when that session is loaded.
- `ensure_geometry_library()` is gone; nothing seeds a geometry into the home
  folder at start-up any more.
