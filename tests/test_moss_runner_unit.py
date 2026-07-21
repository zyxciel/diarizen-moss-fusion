"""Unit tests for MOSS runner helpers (no HF / GPU / from_pretrained)."""
from dataclasses import dataclass
from pathlib import Path

from fusion_diarize.types import AsrStatus, ChunkWindow, Source, Turn


@dataclass
class _Seg:
    start: float
    end: float
    speaker: str
    text: str


def test_segments_to_turns_applies_offset():
    from fusion_diarize.moss_runner import segments_to_turns

    segs = [
        _Seg(0.5, 1.2, "S01", "hello"),
        _Seg(2.0, 3.0, "S02", ""),
    ]
    turns = segments_to_turns(segs, offset=100.0)
    assert len(turns) == 2
    assert turns[0].start == 100.5
    assert turns[0].end == 101.2
    assert turns[0].speaker_id == "S01"
    assert turns[0].text == "hello"
    assert turns[0].asr_status == AsrStatus.PROVISIONAL
    assert turns[0].source == Source.MOSS
    assert turns[1].start == 102.0
    assert turns[1].end == 103.0
    assert turns[1].speaker_id == "S02"
    assert turns[1].text == ""
    assert turns[1].asr_status == AsrStatus.EMPTY
    assert turns[1].source == Source.MOSS


def test_segments_to_turns_speaker_prefix():
    from fusion_diarize.moss_runner import segments_to_turns

    segs = [_Seg(0.0, 1.0, "S01", "a")]
    turns = segments_to_turns(segs, offset=10.0, speaker_prefix="c002:")
    assert turns[0].speaker_id == "c002:S01"
    assert turns[0].start == 10.0


def test_fake_moss_runner_run_chunks_ok_meta(tmp_path: Path):
    from fusion_diarize.moss_runner import FakeMossRunner

    preset = [
        Turn(
            1.0,
            2.0,
            "S01",
            text="hi",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        )
    ]
    runner = FakeMossRunner(turns_by_default=preset)
    chunks = [
        ChunkWindow(0.0, 30.0, high_speaker_density=False),
        ChunkWindow(25.0, 55.0, high_speaker_density=True),
    ]
    audio = tmp_path / "full.wav"
    audio.write_bytes(b"")
    work = tmp_path / "moss"
    turns, meta = runner.run_chunks(audio, chunks, work)

    assert turns == preset
    assert len(meta) == 2
    assert all(m["ok"] is True for m in meta)
    assert meta[0]["chunk"] == 0
    assert meta[0]["start"] == 0.0
    assert meta[0]["end"] == 30.0
    assert meta[1]["chunk"] == 1
    assert meta[1]["high_speaker_density"] is True
    assert work.is_dir()


def test_detect_incomplete_low_coverage():
    from fusion_diarize.moss_runner import detect_incomplete

    turns = [Turn(0.0, 100.0, "c000:S01", text="x")]
    incomplete, reason = detect_incomplete(turns, 0.0, 1200.0)
    assert incomplete is True
    assert "low_coverage" in reason


def test_detect_incomplete_hit_max_tokens():
    from fusion_diarize.moss_runner import detect_incomplete

    turns = [Turn(0.0, 1100.0, "c000:S01", text="x")]
    incomplete, reason = detect_incomplete(
        turns, 0.0, 1200.0, generated_tokens=16384, max_new_tokens=16384
    )
    assert incomplete is True
    assert reason == "hit_max_new_tokens"


def test_detect_complete_ok():
    from fusion_diarize.moss_runner import detect_incomplete

    turns = [Turn(0.0, 1150.0, "c000:S01", text="x")]
    incomplete, reason = detect_incomplete(turns, 0.0, 1200.0)
    assert incomplete is False
    assert reason == ""


def test_moss_runner_module_imports_without_loading_model():
    """Importing the module must not load transformers / HF models."""
    import fusion_diarize.moss_runner as mod

    assert hasattr(mod, "MossRunner")
    assert hasattr(mod, "FakeMossRunner")
    assert hasattr(mod, "segments_to_turns")
    assert hasattr(mod, "detect_incomplete")
