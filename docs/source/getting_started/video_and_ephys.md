(target-video-and-ephys)=
# Video + ephys through neuroconv

When a recording system logged the camera, the video belongs on the recording
clock, and [neuroconv](https://neuroconv.readthedocs.io) is the tool that puts
it there. This page walks one rig through end to end in the shape of the
notebook the {doc}`Data wizard <../advanced/data_wizard>` writes (modes 2 and
3). The recipes themselves are neuroconv's, from its
[how-to on aligning external video](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html);
this page only shows how they meet EthoGraph.

```{tip}
neuroconv is not part of the `ethograph` environment. Install it with
`pip install "neuroconv[intan,video]"` (swap the extras for your recording system).
```

## One rig, end to end

The rig: an Intan recorder writing `session.rhd`, one free-running camera whose
frame-out pin is wired to `DIGITAL-IN-02`, and a trial trigger on
`DIGITAL-IN-01`. The camera reports every exposure, so each frame lands on the
Intan clock and drift is corrected for free.

### 1. Parameters

The first cell of the notebook is tagged `parameters`. `session_dir` is the one
thing to change for the next session on the same rig.

```python
session_dir = "D:/data/rig_A/2026-09-11"
rhd_file = f"{session_dir}/ephys/session.rhd"
nwbfile_path = f"{session_dir}/session.nwb"
```

### 2. Find the media

{func}`~ethograph.discover_media` builds the pairing table the same way it
does in mode 1. For a free-running camera the table has one row; for a
triggered camera it has one row per trial file.

```python
import ethograph as eto

sources = [
    eto.SourceSpec("video", device="cam-1", folder="video"),
    eto.SourceSpec("pose", device="cam-1", folder="pose"),
    eto.SourceSpec("audio", device="mic-1", folder="audio"),
]
trial_table = eto.discover_media(session_dir, sources)
video_files = [f"{session_dir}/video/{name}" for name in trial_table["video_cam-1"]]
```

### 3. Read the pulses off the recorder

```python
from neuroconv.datainterfaces import IntanDigitalInterface, IntanRecordingInterface

recording_interface = IntanRecordingInterface(file_path=rhd_file)
digital_interface = IntanDigitalInterface(
    file_path=rhd_file,
    detection_configuration={
        "DIGITAL-IN-02": [
            {
                "signal_conditioning": {"binarize": "midpoint"},
                "detection": "rising",
                "event_name": "camera_frame",
            }
        ],
        "DIGITAL-IN-01": [
            {
                "signal_conditioning": {"binarize": "midpoint"},
                "detection": "rising",
                "event_name": "trial_trigger",
            }
        ],
    },
)

frame_pulse_times = digital_interface.get_event_times("camera_frame")
trial_onsets = digital_interface.get_event_times("trial_trigger")
```

### 4. Place the frames

```python
from neuroconv.datainterfaces import ExternalVideoInterface

video_interface = ExternalVideoInterface(file_paths=video_files, video_name="video_cam-1")

n_frames = sum(video_interface.get_header_frame_counts())
assert len(frame_pulse_times) == n_frames, (
    f"{len(frame_pulse_times)} pulses for {n_frames} frames"
)
video_interface.alignment["session"].set_times(frame_pulse_times)
```

The assertion is the whole point of the wiring. A few pulses short means the
camera dropped frames; a few too many means the recorder started before the
camera. Never trim the pulses to fit. neuroconv raises on a mismatch too, and
says by how much.

### 5. Write `session.nwb`

```python
from neuroconv import ConverterPipe

converter = ConverterPipe(
    data_interfaces={
        "ephys": recording_interface,
        "events": digital_interface,
        "video_cam-1": video_interface,
    }
)
metadata = converter.get_metadata()
metadata["NWBFile"]["session_description"] = "rig A"

nwbfile = converter.create_nwbfile(metadata=metadata)
trial_duration = 5.0
for onset in trial_onsets:
    nwbfile.add_trial(start_time=onset, stop_time=onset + trial_duration)

converter.run_conversion(nwbfile=nwbfile, nwbfile_path=nwbfile_path, metadata=metadata)
```

The trials table is what turns one session into trials in the GUI. For a
triggered camera the how-to's recipe reads the duration of each trial off its
own file instead of a constant.

### 6. Pair the rest over the `.nwb`

Pose and audio share the camera's clock, so they need no pulses. Pass the
neuroconv file as `output_path` and {func}`~ethograph.pair_media` adds the
streams to it in place, next to the trials it already holds:

```python
eto.pair_media(
    trial_table,
    stream_rates={"pose": 30.0, "audio": 48000.0},
    output_path=nwbfile_path,
    media_root=session_dir,
)
```

### 7. Open it

On the start page, select `session.nwb` in the **Custom set-up** card and click
**Load**. The ephys trace, the video and the pose overlay all read the same
clock.

## Which recipe

The how-to names four ways a camera can be wired to a recorder, and gives one
recipe per (camera, wiring) pair. They are not reproduced here; each link goes
to the recipe.

| Wiring | What is on the sync line | Recipe |
|---|---|---|
| **The camera reports** | A pulse per frame from the camera's strobe or frame-out pin | [pulse per frame](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html#a-free-running-camera): the rig above. Best accuracy; corrects drift and finds dropped frames |
| **The camera is commanded** | A trigger from the recorder to the camera, once per session or per trial | [known offset](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html#a-free-running-camera) or [trial onsets only](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html#a-triggered-camera-one-file-per-trial). The exposure delay is unmeasured |
| **A shared sync source** | A third device pulsing both camera and recorder | [remap times](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html#a-free-running-camera): the camera keeps its own clock and is mapped onto the recorder's |
| **No cable** | Nothing; only a written start time | [known offset](https://neuroconv--2037.org.readthedocs.build/en/2037/how_to/align_external_video.html#a-free-running-camera): corrects the start and nothing else |

A free-running camera split into several files by the recorder, and a
triggered camera with a pulse per frame, each have their own variant on the
same page. The wizard's timing page picks among these for you.

## What rides along

- **Pose.** neuroconv's `DeepLabCutInterface` (and the SLEAP and LightningPose
  ones) write ndx-pose into the same file. Give it the video as `source_video`
  and EthoGraph pairs the pose to that camera. Or leave the pose file beside the
  `.nwb` and pair it with {func}`~ethograph.pair_media`, as in step 6; both
  end up on the same overlay.
- **Spike sorting.** `KiloSortSortingInterface` writes the units, which gives
  you the raster. The {ref}`Phy-like viewer <target-ephys-viewers>` reads
  waveforms straight off the raw binary and still needs the Kilosort folder
  selected in the GUI.

## What does not

- **Audio.** neuroconv's `AudioInterface` embeds the samples in the `.nwb`, and
  EthoGraph plays audio from files. Keep the WAV beside the `.nwb` and pair it
  (step 6). The ephys recording itself is read from its own file too; see
  {doc}`loading_ephys`.

## Paths

`ExternalVideoInterface` stores `external_file` as you gave it, absolute or
relative. EthoGraph does not depend on it staying valid: the media folder you
pick at load time resolves every stream by basename, so a session moved to
another machine opens once you point the GUI at its video folder.
