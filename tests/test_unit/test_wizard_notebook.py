"""Unit tests for the Data wizard's notebook generator (Qt-free)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
import pytest

from ethograph.gui.wizard_notebook import RigSource, RigSpec, build_notebook, write_notebook


def _code_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _all_code(nb: nbformat.NotebookNode) -> str:
    return "\n".join(_code_sources(nb))


def _assert_all_code_cells_parse(nb: nbformat.NotebookNode) -> None:
    for source in _code_sources(nb):
        ast.parse(source)


def _pair_spec() -> RigSpec:
    return RigSpec(
        rig_name="rig_pair",
        session_dir="/data/session1",
        mode="pair",
        sources=[
            RigSource(
                stream="video",
                folder="videos",
                pattern=r"cam(?P<camera>\d+)_trial(?P<trial>\d+)",
                n_devices=2,
            ),
            RigSource(stream="audio", folder="audio", rate=48000.0),
        ],
    )


def _free_running_offset_spec() -> RigSpec:
    return RigSpec(
        rig_name="rig_offset",
        session_dir="/data/session2",
        mode="free_running",
        sources=[
            RigSource(stream="video", folder="videos", n_devices=1),
            RigSource(stream="audio", folder="audio", n_devices=1, rate=48000.0),
        ],
        timing="offset",
        offset_s=0.25,
        recording_file="ephys/session.rhd",
    )


def _free_running_pulses_spec(*, split_files: bool) -> RigSpec:
    return RigSpec(
        rig_name="rig_pulses_free",
        session_dir="/data/session3",
        mode="free_running",
        sources=[RigSource(stream="video", folder="videos", n_devices=2)],
        timing="pulses",
        split_files=split_files,
        recording_file="ephys/session.rhd",
        frame_line="DIGITAL-IN-02",
    )


def _triggered_pulses_spec() -> RigSpec:
    return RigSpec(
        rig_name="rig_pulses_triggered",
        session_dir="/data/session4",
        mode="triggered",
        sources=[RigSource(stream="video", folder="videos", n_devices=1)],
        timing="pulses",
        recording_file="ephys/session.rhd",
        frame_line="DIGITAL-IN-02",
        burst_gap=10.0,
    )


def _triggered_onsets_spec() -> RigSpec:
    return RigSpec(
        rig_name="rig_onsets",
        session_dir="/data/session5",
        mode="triggered",
        sources=[RigSource(stream="video", folder="videos", n_devices=1)],
        timing="onsets",
        recording_file="ephys/session.rhd",
        trigger_line="DIGITAL-IN-01",
    )


class TestPair:
    def test_validates(self) -> None:
        nb = build_notebook(_pair_spec())
        nbformat.validate(nb)

    def test_parameters_cell_first_and_tagged(self) -> None:
        nb = build_notebook(_pair_spec())
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells[0].metadata.get("tags") == ["parameters"]
        assert "session_dir = " in code_cells[0].source

    def test_no_timing_cell(self) -> None:
        nb = build_notebook(_pair_spec())
        headings = [c.source for c in nb.cells if c.cell_type == "markdown"]
        assert not any("Timing" in h for h in headings)

    def test_no_neuroconv_import(self) -> None:
        nb = build_notebook(_pair_spec())
        assert "neuroconv" not in _all_code(nb)

    def test_pattern_is_raw_string_re_compiles(self) -> None:
        nb = build_notebook(_pair_spec())
        params_source = _code_sources(nb)[0]
        match = re.search(r'"pattern":\s*(r?"[^"]*")', params_source)
        assert match is not None
        literal = match.group(1)
        assert literal.startswith('r"'), f"expected a raw string literal, got {literal!r}"
        pattern_value = ast.literal_eval(literal)
        re.compile(pattern_value)  # must not raise

    def test_all_code_cells_parse(self) -> None:
        nb = build_notebook(_pair_spec())
        _assert_all_code_cells_parse(nb)

    def test_pair_media_call_present(self) -> None:
        nb = build_notebook(_pair_spec())
        assert "eto.pair_media(" in _all_code(nb)


class TestFreeRunningOffset:
    def test_validates(self) -> None:
        nb = build_notebook(_free_running_offset_spec())
        nbformat.validate(nb)

    def test_has_timing_cell_with_shift_times(self) -> None:
        nb = build_notebook(_free_running_offset_spec())
        assert "shift_times" in _all_code(nb)
        assert "session_wide" in _all_code(nb)

    def test_uses_neuroconv_converter_pipe(self) -> None:
        nb = build_notebook(_free_running_offset_spec())
        assert "ConverterPipe" in _all_code(nb)

    def test_no_np_split(self) -> None:
        nb = build_notebook(_free_running_offset_spec())
        assert "np.split" not in _all_code(nb)

    def test_all_code_cells_parse(self) -> None:
        nb = build_notebook(_free_running_offset_spec())
        _assert_all_code_cells_parse(nb)


class TestFreeRunningPulses:
    def test_validates_both_variants(self) -> None:
        for split_files in (False, True):
            nb = build_notebook(_free_running_pulses_spec(split_files=split_files))
            nbformat.validate(nb)

    def test_start_at_only_when_split(self) -> None:
        not_split = _all_code(build_notebook(_free_running_pulses_spec(split_files=False)))
        split = _all_code(build_notebook(_free_running_pulses_spec(split_files=True)))
        assert "start_at" not in not_split
        assert "start_at" in split

    def test_no_np_split_in_either_variant(self) -> None:
        for split_files in (False, True):
            nb = build_notebook(_free_running_pulses_spec(split_files=split_files))
            assert "np.split" not in _all_code(nb)

    def test_assert_matches_frame_counts(self) -> None:
        nb = build_notebook(_free_running_pulses_spec(split_files=False))
        assert "get_header_frame_counts" in _all_code(nb)

    def test_all_code_cells_parse(self) -> None:
        for split_files in (False, True):
            nb = build_notebook(_free_running_pulses_spec(split_files=split_files))
            _assert_all_code_cells_parse(nb)


class TestTriggeredPulses:
    def test_validates(self) -> None:
        nb = build_notebook(_triggered_pulses_spec())
        nbformat.validate(nb)

    def test_np_split_present(self) -> None:
        nb = build_notebook(_triggered_pulses_spec())
        assert "np.split" in _all_code(nb)

    def test_no_start_at(self) -> None:
        nb = build_notebook(_triggered_pulses_spec())
        assert "start_at" not in _all_code(nb)

    def test_all_code_cells_parse(self) -> None:
        nb = build_notebook(_triggered_pulses_spec())
        _assert_all_code_cells_parse(nb)


class TestTriggeredOnsets:
    def test_validates(self) -> None:
        nb = build_notebook(_triggered_onsets_spec())
        nbformat.validate(nb)

    def test_start_at_present(self) -> None:
        nb = build_notebook(_triggered_onsets_spec())
        assert "start_at" in _all_code(nb)

    def test_no_np_split(self) -> None:
        nb = build_notebook(_triggered_onsets_spec())
        assert "np.split" not in _all_code(nb)

    def test_trial_metadata_added(self) -> None:
        nb = build_notebook(_triggered_onsets_spec())
        code = _all_code(nb)
        assert "add_trial_column" in code
        assert "add_trial(" in code

    def test_all_code_cells_parse(self) -> None:
        nb = build_notebook(_triggered_onsets_spec())
        _assert_all_code_cells_parse(nb)


class TestWriteNotebook:
    def test_round_trips(self, tmp_path: Path) -> None:
        out_path = tmp_path / "rig.ipynb"
        result = write_notebook(_pair_spec(), out_path)
        assert result == out_path
        assert out_path.exists()
        nb = nbformat.read(str(out_path), as_version=4)
        nbformat.validate(nb)
        assert nb.cells[1].metadata.get("tags") == ["parameters"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out_path = tmp_path / "nested" / "rig.ipynb"
        write_notebook(_pair_spec(), out_path)
        assert out_path.exists()


class TestTimingRequiresVideoSource:
    def test_raises_without_video(self) -> None:
        spec = RigSpec(
            rig_name="rig_no_video",
            session_dir="/data/session6",
            mode="triggered",
            sources=[RigSource(stream="audio", folder="audio")],
            timing="onsets",
            recording_file="ephys/session.rhd",
            trigger_line="DIGITAL-IN-01",
        )
        with pytest.raises(ValueError, match="no video source"):
            build_notebook(spec)
