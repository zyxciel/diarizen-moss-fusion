from fusion_diarize.fuse_c import (
    detect_speaker_explosion,
    fuse_mode_c,
)
from fusion_diarize.types import AsrStatus, Source, Turn


def test_detect_speaker_explosion_threshold():
    assert detect_speaker_explosion({f"c000:S{i:02d}" for i in range(30)}, {f"speaker_{i}" for i in range(5)})
    assert not detect_speaker_explosion(
        {f"c000:S{i:02d}" for i in range(5)}, {f"speaker_{i}" for i in range(5)}
    )
    # abs_cap=12 dominates when n_dz is small
    assert detect_speaker_explosion({f"l{i}" for i in range(13)}, {"speaker_0"})
    assert not detect_speaker_explosion({f"l{i}" for i in range(12)}, {"speaker_0"})


def test_mode_c_normal_moss_primary_no_diarizen_intervals():
    diarizen = [Turn(0.0, 10.0, "speaker_0")]
    moss_raw = [Turn(0.1, 4.9, "c000:S01", text="hello", asr_status=AsrStatus.PROVISIONAL)]
    moss_remapped = [
        Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL, source=Source.MOSS)
    ]
    meta = [{"ok": True, "incomplete": False, "start": 0.0, "end": 10.0}]
    fused, info = fuse_mode_c(diarizen, moss_raw, moss_remapped, meta)
    assert info["fusion_path"] == "moss_primary"
    assert info["explosion"] is False
    assert all(t.source != Source.DIARIZEN for t in fused)
    assert len(fused) == 1
    assert fused[0].text == "hello"
    assert abs(fused[0].start - 0.1) < 1e-6


def test_mode_c_incomplete_gapfill():
    diarizen = [Turn(0.0, 200.0, "speaker_0")]
    moss_raw = [Turn(0.0, 100.0, "c000:S01", text="partial", asr_status=AsrStatus.PROVISIONAL)]
    moss_remapped = [
        Turn(0.0, 100.0, "speaker_0", text="partial", asr_status=AsrStatus.PROVISIONAL)
    ]
    meta = [{"ok": True, "incomplete": True, "start": 0.0, "end": 200.0}]
    fused, info = fuse_mode_c(diarizen, moss_raw, moss_remapped, meta)
    assert info["fusion_path"] == "moss_primary_gapfill"
    assert any(t.source in (Source.FUSED, Source.MOSS) and t.end <= 100.1 for t in fused)
    assert any(
        t.source == Source.DIARIZEN and t.start >= 99.9 and t.end >= 199.0 for t in fused
    )
    # No DiariZen under MOSS span
    for t in fused:
        if t.source == Source.DIARIZEN:
            assert t.start >= 99.9


def test_mode_c_explosion_uses_diarizen_backbone():
    diarizen = [
        Turn(0.0, 5.0, "speaker_0"),
        Turn(5.0, 10.0, "speaker_1"),
        Turn(10.0, 15.0, "speaker_2"),
        Turn(15.0, 20.0, "speaker_3"),
        Turn(20.0, 25.0, "speaker_4"),
    ]
    moss_raw = [
        Turn(float(i), float(i) + 0.5, f"c000:S{i:02d}", text=f"t{i}")
        for i in range(30)
    ]
    # Remap many locals onto few globals (many-to-one)
    moss_remapped = [
        Turn(
            float(i),
            float(i) + 0.5,
            f"speaker_{i % 5}",
            text=f"t{i}",
            asr_status=AsrStatus.PROVISIONAL,
        )
        for i in range(30)
    ]
    meta = [{"ok": True, "incomplete": False, "start": 0.0, "end": 30.0}]
    fused, info = fuse_mode_c(diarizen, moss_raw, moss_remapped, meta)
    assert info["explosion"] is True
    assert info["fusion_path"] == "diarizen_backbone_explosion"
    assert info["n_moss_local"] == 30
    assert len([t for t in fused if t.source in (Source.DIARIZEN, Source.FUSED)]) >= 5
    # Backbone intervals match DiariZen starts
    starts = sorted(t.start for t in fused)
    assert starts[0] == 0.0


def test_mode_c_explosion_attaches_moss_text():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss_raw = [Turn(0.0, 5.0, f"c000:S{i:02d}") for i in range(20)]
    moss_remapped = [
        Turn(0.05, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL)
    ]
    meta = [{"ok": True, "incomplete": False}]
    fused, info = fuse_mode_c(diarizen, moss_raw, moss_remapped, meta)
    assert info["explosion"] is True
    assert any(t.text == "hello" for t in fused)


def test_mode_c_dedupes_overlap():
    diarizen = [Turn(0.0, 10.0, "speaker_0")]
    moss_raw = [
        Turn(0.0, 5.0, "c000:S01", text="a"),
        Turn(0.1, 5.1, "c000:S01", text="a2"),
    ]
    moss_remapped = [
        Turn(0.0, 5.0, "speaker_0", text="a", asr_status=AsrStatus.PROVISIONAL),
        Turn(0.1, 5.1, "speaker_0", text="a2", asr_status=AsrStatus.PROVISIONAL),
    ]
    meta = [{"ok": True, "incomplete": False}]
    fused, info = fuse_mode_c(diarizen, moss_raw, moss_remapped, meta)
    assert info["fusion_path"] == "moss_primary"
    assert len(fused) == 1
