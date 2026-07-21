from fusion_diarize.types import Turn
from fusion_diarize.chunk_planner import plan_chunks, DEFAULT_TARGET_MAX, DEFAULT_HARD_CAP


def test_short_audio_single_chunk():
    turns = [Turn(0.0, 10.0, "speaker_0")]
    chunks = plan_chunks(turns, duration=120.0)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 120.0
    assert chunks[0].end - chunks[0].start <= DEFAULT_TARGET_MAX


def test_default_cap_is_20_minutes():
    assert DEFAULT_TARGET_MAX == 1200.0
    assert DEFAULT_HARD_CAP == 1200.0


def test_long_audio_splits_near_gap_under_20min():
    # ~2 hours; silence gap around 1100s (within first 20-min window search)
    turns = [
        Turn(0.0, 1000.0, "speaker_0"),
        Turn(1150.0, 7000.0, "speaker_1"),
    ]
    chunks = plan_chunks(
        turns,
        duration=7200.0,
        target_min=900.0,
        target_max=1200.0,
        hard_cap=1200.0,
        overlap=30.0,
        max_local_speakers=8,
    )
    assert len(chunks) >= 2
    assert all(c.end - c.start <= 1200.0 + 1e-6 for c in chunks)
    # first cut near the gap
    assert any(1000 <= c.end <= 1200 for c in chunks[:-1])


def test_high_speaker_density_flag():
    turns = [Turn(0.0, 100.0, f"speaker_{i}") for i in range(10)]
    chunks = plan_chunks(turns, duration=100.0)
    assert chunks[0].high_speaker_density is True
    assert chunks[0].n_local_speakers >= 9
