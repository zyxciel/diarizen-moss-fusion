from fusion_diarize.types import Turn
from fusion_diarize.chunk_planner import plan_chunks


def test_short_audio_single_chunk():
    turns = [Turn(0.0, 10.0, "speaker_0")]
    chunks = plan_chunks(turns, duration=120.0, target_min=1800, target_max=3600, hard_cap=5400)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 120.0


def test_long_audio_splits_near_gap():
    turns = [
        Turn(0.0, 3400.0, "speaker_0"),
        Turn(3600.0, 7000.0, "speaker_1"),
    ]
    chunks = plan_chunks(
        turns, duration=7200.0,
        target_min=1800, target_max=3600, hard_cap=5400,
        overlap=30.0, max_local_speakers=8,
    )
    assert len(chunks) >= 2
    assert all(c.end - c.start <= 5400 + 1e-6 for c in chunks)
    assert any(3300 <= c.end <= 3700 for c in chunks[:-1])


def test_high_speaker_density_flag():
    turns = [Turn(0.0, 100.0, f"speaker_{i}") for i in range(10)]
    chunks = plan_chunks(turns, duration=100.0, target_min=1800, target_max=3600, hard_cap=5400)
    assert chunks[0].high_speaker_density is True
    assert chunks[0].n_local_speakers >= 9
