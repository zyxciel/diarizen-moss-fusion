from fusion_diarize.types import Turn, AsrStatus, Source
from fusion_diarize.fuse_a import fuse_mode_a, dedupe_overlapping_turns
from fusion_diarize.fuse_b import fuse_mode_b


def test_mode_a_prefers_moss_boundaries_when_confident():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [
        Turn(
            0.1,
            4.9,
            "speaker_0",
            text="hello",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        )
    ]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.9}, tau=0.6)
    moss_fused = [t for t in fused if t.source == Source.FUSED]
    assert len(moss_fused) == 1
    assert abs(moss_fused[0].start - 0.1) < 1e-6
    assert moss_fused[0].text == "hello"
    # Tiny DiariZen edges outside MOSS are OK (gap fill), not dual mid-span copies
    for t in fused:
        if t.source == Source.DIARIZEN:
            assert t.end <= 0.1 + 1e-6 or t.start >= 4.9 - 1e-6


def test_mode_a_fallback_diarizen_on_low_conf():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [
        Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL)
    ]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.2}, tau=0.6)
    assert fused[0].start == 0.0
    assert fused[0].source == Source.DIARIZEN


def test_mode_a_fills_diarizen_gaps_when_moss_incomplete():
    """Incomplete MOSS must not wipe the rest of the DiariZen track (miss)."""
    diarizen = [Turn(0.0, 100.0, "speaker_0")]
    moss = [
        Turn(
            0.0,
            40.0,
            "speaker_0",
            text="partial",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        )
    ]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.9}, tau=0.6)
    assert any(t.source == Source.FUSED and t.end <= 40.1 for t in fused)
    # DiariZen fill after MOSS ends
    assert any(
        t.source == Source.DIARIZEN and t.start >= 39.9 and t.end >= 99.0 for t in fused
    )


def test_mode_a_dedupes_overlap_and_avoids_dual_hypothesis():
    """Chunk-overlap MOSS copies + DiariZen must not both survive (FA)."""
    diarizen = [Turn(0.0, 10.0, "speaker_0")]
    moss = [
        Turn(0.0, 5.0, "speaker_0", text="a", asr_status=AsrStatus.PROVISIONAL),
        Turn(0.1, 5.1, "speaker_0", text="a2", asr_status=AsrStatus.PROVISIONAL),  # overlap dup
    ]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.9}, tau=0.6)
    moss_like = [t for t in fused if t.source == Source.FUSED]
    assert len(moss_like) == 1
    # No DiariZen under the MOSS span
    for t in fused:
        if t.source == Source.DIARIZEN:
            assert t.start >= 4.9


def test_mode_a_drops_unmapped_moss_overlapping_diarizen():
    diarizen = [Turn(0.0, 10.0, "speaker_0")]
    moss = [
        Turn(1.0, 3.0, "c000:S99", text="ghost", asr_status=AsrStatus.PROVISIONAL),
    ]
    fused = fuse_mode_a(diarizen, moss, confidences={}, tau=0.6)
    assert all(t.speaker_id != "c000:S99" for t in fused)
    assert any(t.source == Source.DIARIZEN for t in fused)


def test_dedupe_overlapping_turns():
    turns = [
        Turn(0.0, 5.0, "speaker_0"),
        Turn(0.2, 5.1, "speaker_0"),
        Turn(6.0, 7.0, "speaker_0"),
    ]
    out = dedupe_overlapping_turns(turns, iou_thresh=0.5)
    assert len(out) == 2


def test_mode_b_voting_keeps_speech():
    diarizen = [Turn(0.0, 2.0, "speaker_0")]
    moss = [Turn(1.0, 3.0, "speaker_0", text="x", asr_status=AsrStatus.PROVISIONAL)]
    fused = fuse_mode_b(diarizen, moss, frame_hop=0.2)
    assert any(t.speaker_id == "speaker_0" for t in fused)
    assert max(t.end for t in fused) >= 2.5
