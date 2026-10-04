# Bundled defaults

The project folder every install starts from. On GUI launch each file here is
copied into `~/.ethograph/defaults/` **if it is not there yet**
(`ethograph.utils.paths.seed_defaults`); a file the user edited is never
overwritten, and a file they deleted comes back — disable a geometry by not
selecting it, not by deleting it. `~/.ethograph/defaults/` has the shape of a
project folder and stands in for one while no project is chosen on the cover
page.

```
mapping.txt             # the label vocabulary the GUI falls back on
segment/segment.yaml    # action segmentation — copy into a project and edit
spot/spot.yaml          # pixel event spotting — same
```

## Reference geometries (`space/`)

Not shipped here: the Space plot's reference geometries are a library like the
skeleton's, read nearest first from the session's `.ethograph/space/`, the
project's `space/` and the starter project's `~/.ethograph/defaults/space/`
(`plots_space.geometry_dirs`). The Moll et al., 2025 template writes its arena
(`ethograph/assets/space/moll2025.yaml`) into its own session. Each `*.yaml` is
one selectable geometry, identified by its filename (without extension); all
`references` in the file are drawn together.

Pick the one drawn under **Data → Space controls → "Library geometry:"**, or
set it as a default in `gui_settings.yaml` (global) or a dataset's
`local_settings.yaml` (per dataset):

```yaml
space_library_geometry: moll2025
```

Schema:

```yaml
references:
  - name: setup            # label used in log messages only
    vertices:              # [x, y] or [x, y, z] points
      - [0.0, 0.0, 0.0]
      - [1.0, 0.0, 0.0]
    edges:                 # pairs of vertex indices
      - [0, 1]
    color: black           # any pyqtgraph color name/hex
```

## Contributing

Add a file to this folder via PR to ship it as a default. The example configs
must keep loading: `tests/test_unit/test_home_layout.py` builds each one.
