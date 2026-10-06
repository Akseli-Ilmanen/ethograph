# 3D from several cameras

Two or more calibrated cameras that see the same keypoint place it in 3D.
Ethograph does this with [aniposelib](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html):
every calibration, triangulation and reprojection is aniposelib's own
function, and the calibration file is Anipose's own `calibration.toml`, so it
stays usable in Anipose itself. Ethograph's part is reading the 2D points —
whatever tool tracked them — and writing the result where the rest of the GUI
can use it.

| You have | Do this |
|---|---|
| A DeepLabCut 3D project, already calibrated | {ref}`Import its calibration <target-import-dlc-calibration>`, once per rig |
| A new rig, or more than two cameras | {ref}`Calibrate with aniposelib <target-new-calibration>` |
| 2D pose files per camera (DeepLabCut, SLEAP, LightningPose, …) | {ref}`Triangulate the session <target-triangulate-session>` |
| No pose files — you label by hand | {doc}`Label in two views <keypoint_labelling/two_views>` |

## The calibration lives in the project

```
my_project/
    └── calibration/
        ├── rig.toml                   # aniposelib's calibration.toml: one entry per camera
        └── rig.frame.yaml             # optional: the world frame for that rig
```

A calibration describes a rig, not a recording, so it has one home: the
project folder's `calibration/`. **A project with one calibration needs no
choosing.** With several — a rig that was moved in March — the GUI asks once
per dataset which one it was filmed with and remembers the answer; a script
names it (`calibration="rig_2025-03"`).

The camera names in the file must be the session's camera names. A
calibration naming a camera the session does not have is refused, by name.

(target-import-dlc-calibration)=
## From a DeepLabCut 3D project

**Tools ▸ 3D: Import DeepLabCut calibration…**, then pick the DeepLabCut 3D
project folder (the one holding `config.yaml` and `camera_matrix/`). Or:

```python
from ethograph import triangulate as tri

tri.import_dlc_calibration("DLC/3D_project", "my_project")   # → my_project/calibration/3D_project.toml
```

The camera parameters are **copied, not re-estimated** — the calibration you
already validated is the one you keep — and the 3D coordinates come out in the
frame DeepLabCut 3D writes, so a session triangulated here lines up with the
`_DLC_3D.h5` files you already have. The DeepLabCut folder is read once and
never again.

DeepLabCut 3D calibrates a camera *pair*; that is the one limit of this route.

(target-new-calibration)=
## A new rig

Calibrate with aniposelib directly and save into the project — there is
nothing for Ethograph to add:

```python
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

board = CharucoBoard(7, 10, square_length=25, marker_length=18.75, marker_bits=4, dict_size=50)
cgroup = CameraGroup.from_names(["cam-1", "cam-2", "cam-3"])    # the session's camera names
cgroup.calibrate_videos([["calib-cam-1.mp4"], ["calib-cam-2.mp4"], ["calib-cam-3.mp4"]], board)
cgroup.dump("my_project/calibration/rig.toml")
```

The unit of `square_length` is the unit your 3D coordinates come out in.

(target-triangulate-session)=
## Triangulating a session

**Tools ▸ 3D: Triangulate poses…** runs over the trials the trials table
currently shows and writes two variables into the session's dataset:

| Variable | Dims | |
|---|---|---|
| `position_3d` | `time, space (x, y, z), keypoint, individual` | the triangulated keypoints — an ordinary feature for the Space plot and the model pipelines |
| `reprojection_error` | `time, keypoint, individual` | pixels between each 3D point, projected back, and the 2D points it came from |

The 2D points are read the way the video overlay reads them: a dataset
variable with a `camera` dimension if the session has one, else the pose file
of each camera. Cameras with different frame rates or start times are paired
by the moment on the trial clock, never by frame index.

Scripted, it is one call, and the place for everything the menu does not ask:

```python
tri.triangulate(
    "my_project/session_01",          # session folder
    "my_project",                     # project folder
    min_confidence=0.9,               # a camera scoring a point lower did not see it
    method="triangulate",             # or "ransac", "optim"
    source_software="DeepLabCut",     # only needed for pose files
)
```

| `method` | aniposelib function | Use it |
|---|---|---|
| `triangulate` | `CameraGroup.triangulate` | the default: fast, linear |
| `ransac` | `CameraGroup.triangulate_ransac` | three or more cameras, when one is sometimes wrong |
| `optim` | `CameraGroup.triangulate_optim` | slow; smooths over time and holds rigid segments rigid |

For `optim`, name the rigid segments as keypoint pairs — `constraints` for
truly rigid ones, `constraints_weak` for nearly rigid ones:

```python
tri.triangulate(..., method="optim",
                constraints=[("beakTip", "beakBase")],
                constraints_weak=[("leye", "reye")])
```

A re-run replaces `position_3d` for the trials it covers and leaves the other
trials alone. Each `position_3d` records the calibration, the method and the
threshold it was made with in its attributes.

(target-world-frame)=
## The world frame

Out of the box the coordinates are in the calibration's frame: a camera's,
tilted against the room, in the calibration board's units. To have x, y and z
mean something, write a `{name}.frame.yaml` beside the calibration, naming two
axes and an origin by keypoint — fixed landmarks your pose files track, such
as the corners of the arena:

```yaml
axes:
  - [x, box1, box2]                  # x runs from box1 to box2
  - [z, box1, box5]                  # z runs from box1 to box5
reference_point: box1                # the origin
scale: {from: box1, to: box2, length: 0.30}   # optional: that edge is 0.30 long…
unit: m                              # …in metres
```

`axes` and `reference_point` are spelled as in Anipose's `config.toml`. The
second axis is made perpendicular to the first, so the frame is rigid even if
the landmarks are not perfectly square.

The first triangulation that sees the landmarks resolves the frame and writes
the result (`rotation`, `origin`, `scale_factor`) back into the file. From
then on every session of that rig lands in the same frame, whether or not its
own videos show the landmarks. Delete those three keys to fit it again.

## The room as a geometry

Landmarks that do not move have one 3D position each — which is exactly what
the Space plot's reference geometry is made of:

```python
tri.export_geometry(
    "my_project/session_01", "my_project", "arena",
    keypoints=["box1", "box2", "box3", "box4"],
)                                    # → my_project/space/arena.yaml
```

The vertices are the keypoints' median `position_3d`; the edges are the
project skeleton's connections between them (or pass `edges=[("box1", "box2"), …]`).
The file is an ordinary geometry: pick it under **Library geometry** in the
Space plot's settings. When labelling by hand, the same export is a button —
see {doc}`keypoint_labelling/two_views`.
