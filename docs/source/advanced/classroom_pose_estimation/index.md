# Classroom pose estimation

**Tools ▸ Pose tracking (from scratch)…** (or **Label keypoints…** in the Pose
sidebar) opens a dialog where you label a few frames by clicking the video and
let a point tracker fill in the rest. No training, no annotated dataset and
**no GPU**: it runs on the laptops students already have. Film a short clip of
an animal on your phone, drop it into Ethograph, and you get from a handful of
clicks to a pose trace within an hour.

Once the frames are filled, load the result into the GUI and velocity, speed
and acceleration sit on the same time axis as the video and the sound, so you
can check the trace against what the animal actually did.
{doc}`../../examples/create_dataset_cricket` does exactly this with cricket
leg kinematics and the chirp they produce.


## The steps

```{toctree}
:maxdepth: 1

labelling
detect
fill
correction
export
two_views
```

