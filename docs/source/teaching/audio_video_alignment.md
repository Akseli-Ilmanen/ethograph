# Aligning a microphone to a video

A camera records its own soundtrack; a better microphone beside it was
started by hand, a few seconds earlier or later. Both heard the same room, so
the two recordings can be laid over each other and the lag at which they
agree is the offset the microphone's file needs to play in step with the
video. Ethograph proposes that offset, and which audio file belongs to which
video, and you approve each one on a spectrogram before anything is written.

Open it from the start page: **🛠 More tools → GUI tools → Pair & align audio
with video…**. It needs the audio extra (`pip install ethograph[audio]`).

## What it needs

- **As many audio files as videos.** One recording per trial on each side;
  unequal counts are refused. A single microphone file spanning several
  videos is not handled here (pair it as a session-wide stream, see
  {doc}`../getting_started/your_data/media_alignment`).
- **Videos with a soundtrack.** The camera's own audio is the reference. A
  silent video has nothing to compare against and is named in an error.

## Analyse

Drop (or *Add…*) the videos on the left and the audio files on the right, then
**Analyse**. Each video's soundtrack is extracted once (to the same cache the
GUI uses for playback), and every soundtrack is compared with every audio
file. The comparison works on standardised MFCC frames every 10 ms rather than
the raw waveforms, so the camera's lossy codec and the two microphones'
different frequency responses do not matter; what has to match is *when*
sounds happen.

The table then holds one row per video:

| Column | Meaning |
|---|---|
| **Audio file** | The proposed partner, chosen so that the whole set of matches scores best. Change it from the drop-down; the file it displaces swaps into the row that held the one you picked. A video name in amber means the content match disagrees with the files' sort order. |
| **Offset (s)** | Where the audio file's first sample falls on the video's clock. Negative: the microphone started before the camera. Nudge it by hand if the preview says so. |
| **Score** | How far the chosen correlation peak stands above the rest. Above 10 is a clear match; below 5 (red) means the peak barely stands out and the row needs a careful look or a manual offset. The score grows with the length of the overlap, so short clips score lower than long ones. |
| **Duration** | Both files' lengths, a cheap sanity check on the pairing. |
| **Accept** | Your approval. Every row must be accepted before the session can be created; changing a row's file or offset clears it. |

## Check

Selecting a row shows two spectrograms eight seconds wide, cut around the
loudest moment the two recordings share: the video's soundtrack above, the
microphone below, **already shifted by the offset**. A right offset shows the
same sounds one above the other — a call, a door, a peck landing at the same
x. A wrong one shows them displaced, by exactly the amount the offset is off.
The title spells the result out ("audio starts 5.00 s before video").
The window maximizes like any other, and **F11** toggles full screen for a
close look.

When a file is long enough, the offset is also measured in one-minute windows
along it. If those drift apart the title warns that the two devices' clocks
run at different rates (in parts per million). Ethograph stores one offset per
file, so with a large drift the alignment is exact where the loudest moment
was and slides away from it; split the recording into shorter trials if that
matters.

## Create session

**Create session** asks which folder becomes the session (when the files came
from more than one), writes its `.ethograph/alignment.nwb` and loads it. The
result is an ordinary multi-trial session, one trial per video:

- `video_cam-1` — the video.
- `audio_mic-1` — your audio file, placed by its own start on the session
  clock (`audio_mic-1_start`), its full length kept.
- `audio_cam-1-track` — the camera's soundtrack, copied into
  `.ethograph/video_audio/video-track_{video}.wav` so its name says what it is
  in the audio panels. Keep it as the sanity check it is, or delete the folder.

The session gets a drop record, so it reopens from the start page like any
drag & drop.
