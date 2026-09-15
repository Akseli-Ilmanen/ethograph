"""Streaming a video into the model must be the folder path without the folder.

What earns a test: the window starts must be the ones the vendored
``ActionSpotVideoDataset`` enumerates (padding at the head, the tail rule);
the streamed per-frame scores must match what the vendored reader produces
from an exported folder of the same video; and a frame must be the export's
frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ethograph.io.video_decode import iter_rgb_frames
from ethograph.spot.dataset import TrialRecord, export_frames
from ethograph.spot.e2espot.dataset.frame import ActionSpotVideoDataset
from ethograph.spot.e2espot.train_e2e import E2EModel
from ethograph.spot.stream import (
    GraphedForward,
    eager_forward,
    load_run_model,
    predict_trial,
    prepare_frame,
    window_starts,
)


def _write_video(path: Path, n_frames: int, size=(64, 48), fps=25) -> None:
    av = pytest.importorskip("av")
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            frame = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
            frame[:, : 8 + i % 20] = 255  # something that moves, so windows differ
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _record(tmp_path, n_frames=60, crop=None) -> TrialRecord:
    video = tmp_path / "v.mp4"
    _write_video(video, n_frames)
    return TrialRecord(
        video_id="v",
        source=tmp_path / "s.nc",
        trial=1,
        video_path=video,
        num_frames=n_frames,
        fps=25.0,
        width=40,
        height=32,
        events={},
        crop=crop,
    )


def _fake_run(tmp_path, clip_len=8, stride=2, crop_dim=32) -> Path:
    """A tiny untrained bw model saved the way the trainer saves one."""
    import torch

    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    stored = {
        "feature_arch": "rny002",
        "temporal_arch": "gru",
        "clip_len": clip_len,
        "stride": stride,
        "dilate_len": 1,
        "modality": "bw",
        "crop_dim": crop_dim,
        "dataset": "x",
    }
    (run_dir / "config.json").write_text(json.dumps(stored))
    torch.manual_seed(0)
    model = E2EModel(3, "rny002", "gru", clip_len=clip_len, modality="bw", device="cpu")
    torch.save(model.state_dict(), run_dir / "checkpoint_000.pt")
    return run_dir


class TestWindows:
    @pytest.mark.parametrize("num_frames,clip_len,stride", [(60, 8, 2), (1000, 200, 2), (7, 8, 1), (200, 200, 2)])
    def test_starts_are_the_vendored_datasets(self, tmp_path, num_frames, clip_len, stride):
        labels = tmp_path / "split.json"
        labels.write_text(json.dumps([{"video": "v", "num_frames": num_frames, "fps": 25.0, "events": []}]))
        ds = ActionSpotVideoDataset(
            {"a": 1}, str(labels), str(tmp_path), "bw", clip_len, overlap_len=clip_len // 2, stride=stride
        )
        assert window_starts(num_frames, clip_len, stride, clip_len // 2) == [s for _, s in ds._clips]


class TestDecode:
    def test_rgb_frames_are_pyavs_own(self, tmp_path):
        """One conversion context kept for the whole video must give PyAV's per-frame pixels."""
        av = pytest.importorskip("av")
        record = _record(tmp_path, n_frames=12)
        ours = list(iter_rgb_frames(record.video_path))
        with av.open(str(record.video_path)) as container:
            theirs = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
        assert len(ours) == 12
        assert all(np.array_equal(a, b) for a, b in zip(ours, theirs))
        assert len(list(iter_rgb_frames(record.video_path, step=5))) == 3


class TestGraphedForward:
    @pytest.mark.filterwarnings("ignore::FutureWarning")  # the vendored predict's torch.cuda.amp.autocast
    def test_replay_is_the_eager_forward(self, tmp_path):
        """A captured graph must give the eager scores, on the first input and on every input after it."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        run_dir = _fake_run(tmp_path)
        model, _ = load_run_model(run_dir, 0, 2, "cuda")
        seq = torch.rand(2, 8, 1, 32, 32, device="cuda")
        eager = eager_forward(model)(seq, None)
        graphed = GraphedForward(model)
        np.testing.assert_allclose(graphed(seq, None), eager, atol=1e-3)
        np.testing.assert_allclose(graphed(seq.flip(0), None), eager[::-1], atol=1e-3)
        assert len(graphed._graphs) == 1


class TestFrame:
    def test_prepare_is_crop_then_resize_then_jpeg(self, tmp_path):
        record = _record(tmp_path, n_frames=2, crop=(10, 5, 50, 45))
        raw = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        out = prepare_frame(raw, record, jpeg_roundtrip=False)
        assert out.shape == (32, 40, 3)
        again = prepare_frame(raw, record, jpeg_roundtrip=True)
        assert again.shape == out.shape and not np.array_equal(again, out)  # JPEG left its mark


class TestStreamMatchesTheFolder:
    def test_scores_agree_with_the_vendored_reader(self, tmp_path):
        import torch

        record = _record(tmp_path, n_frames=60, crop=(4, 2, 60, 46))
        run_dir = _fake_run(tmp_path)
        model, stored = load_run_model(run_dir, 0, 2, "cpu")
        streamed, total = predict_trial(model, stored, record, jpeg_roundtrip=True)
        assert total == 60 and streamed.shape == (30, 3)

        # the folder path: export the frames, read them with the vendored dataset
        frames_dir = tmp_path / "frames"
        export_frames(record, frames_dir)
        labels = tmp_path / "split.json"
        labels.write_text(json.dumps([{"video": "v", "num_frames": 60, "fps": 25.0, "events": []}]))
        ds = ActionSpotVideoDataset(
            {"a": 1, "b": 2}, str(labels), str(frames_dir), "bw", 8, overlap_len=4, crop_dim=32, stride=2
        )
        acc = np.zeros((30, 3), np.float32)
        support = np.zeros(30, np.int32)
        for item in (ds[i] for i in range(len(ds))):
            _, scores = model.predict(item["frame"], use_amp=False)
            s = scores[0]
            start = item["start"]
            if start < 0:
                s = s[-start:]
                start = 0
            end = min(30, start + len(s))
            acc[start:end] += s[: end - start]
            support[start:end] += 1
        reference = acc / support[:, None]
        np.testing.assert_allclose(streamed, reference, atol=2e-2)
        assert not torch.allclose(torch.from_numpy(streamed), torch.full_like(torch.from_numpy(streamed), 1 / 3))


class _Recorder:
    """A model that only remembers what it was fed; no vendored model needed."""

    device = "cpu"
    _num_classes = 3

    def __init__(self) -> None:
        self.seen: list = []

    def predict(self, seq, use_amp=False, fuse=None):
        import torch

        self.seen.append((seq.clone(), None if fuse is None else fuse.clone()))
        per_frame = seq.float().mean(dim=(2, 3, 4))
        return None, torch.stack([per_frame, per_frame * 2, per_frame * 3], -1).numpy()


class TestRollingBuffer:
    """The one-window buffer must hand every window the frames a full decode would: the stride grid,
    the head/tail padding, the eviction — and only the grid frames, never a neighbour off it."""

    @pytest.mark.parametrize("num_frames,clip_len,stride", [(61, 8, 2), (7, 8, 1), (100, 10, 3)])
    def test_windows_are_cut_from_the_full_decode(self, tmp_path, num_frames, clip_len, stride):
        import torch

        from ethograph.spot.dataset import _iter_frames
        from ethograph.spot.stream import normalise

        record = _record(tmp_path, n_frames=num_frames, crop=(4, 2, 60, 46))
        stored = {"clip_len": clip_len, "stride": stride, "crop_dim": 24, "modality": "rgb"}
        block = np.arange(num_frames // stride * 2, dtype=np.float32).reshape(-1, 2)
        model = _Recorder()
        scores, total = predict_trial(model, stored, record, block=block, batch_frames=3 * clip_len)
        assert total == num_frames

        # the reference: decode everything, build each window by hand
        prepared = [prepare_frame(f, record) for f in _iter_frames(record.video_path)]
        seen = [(s, f) for batch, fb in model.seen for s, f in zip(batch, fb)]
        starts = window_starts(num_frames, clip_len, stride, clip_len // 2)
        assert len(seen) == len(starts)
        for start, (seq, fuse) in zip(starts, seen):
            grid = list(range(start, start + clip_len * stride, stride))
            inside = [i for i in grid if 0 <= i < num_frames]
            expected = normalise(torch.from_numpy(np.stack([prepared[i] for i in inside])), 24, "rgb")
            head, tail = sum(i < 0 for i in grid), sum(i >= num_frames for i in grid)
            expected = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, 0, head, tail))
            assert torch.equal(seq, expected)
            rows = np.zeros((clip_len, 2), np.float32)
            for k, i in enumerate(grid):
                if 0 <= i // stride < len(block) and i >= 0:
                    rows[k] = block[i // stride]
            assert np.array_equal(fuse.numpy(), rows)
        assert scores.shape == (num_frames // stride, 3)
