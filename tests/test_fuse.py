from fusion_diarize.types import Turn, AsrStatus, Source
from fusion_diarize.fuse_a import fuse_mode_a
from fusion_diarize.fuse_b import fuse_mode_b


def test_mode_a_prefers_moss_boundaries_when_confident():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL, source=Source.MOSS)]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.9}, tau=0.6)
    assert len(fused) == 1
    assert abs(fused[0].start - 0.1) < 1e-6
    assert fused[0].text == "hello"
    assert fused[0].source == Source.FUSED


def test_mode_a_fallback_diarizen_on_low_conf():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL)]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.2}, tau=0.6)
    assert fused[0].start == 0.0
    assert fused[0].source == Source.DIARIZEN


def test_mode_b_voting_keeps_speech():
    diarizen = [Turn(0.0, 2.0, "speaker_0")]
    moss = [Turn(1.0, 3.0, "speaker_0", text="x", asr_status=AsrStatus.PROVISIONAL)]
    fused = fuse_mode_b(diarizen, moss, frame_hop=0.2)
    assert any(t.speaker_id == "speaker_0" for t in fused)
    assert max(t.end for t in fused) >= 2.5
