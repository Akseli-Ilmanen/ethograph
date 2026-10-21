# 1-2. Define keypoints and label frames

## 1. Define keypoints

Name the body parts you want to track (beak, left leg, …) and, if there is
more than one animal, the individuals. Tick **Static** beside a landmark that
never moves, such as an arena corner: label it once and it is on every frame.

## 2. Label & Edit

1. Arm **Sequential** and click the video once per keypoint. Each click places
   the active keypoint and moves on to the next one. `Tab` skips a keypoint
   you cannot see, `1`–`9` switch individual, `Backspace` deletes,
   `Ctrl+Z` undoes.
2. Press `N` to jump to the next suggested frame and repeat. Labelling every
   frame is wasted effort — neighbouring frames look the same — so aim for
   the default **10 % of frames, roughly every 10th**.
3. **Loop** is the other way round: one keypoint, swept across frames. Use it
   to fix a single body part the fill keeps losing.

A **solid** marker is your label; a **hollow** one is a prediction. Click a
hollow marker to accept it, drag it to correct it.

Tick **Lock** to pan and zoom without placing points. Switching to another
tab locks the pointer for you.

### Which frames to label

| Method | Use it |
|---|---|
| **Evenly spaced** | first pass on any clip |
| **Biggest pixel change** | the animal moves fast and the fill cuts corners |
| **Most different frames** | long clips where the scene changes |
| **Lowest fill confidence** | after a fill — see {ref}`target-correction-loop` |
