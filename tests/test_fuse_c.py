from fusion_diarize.fuse_c import fuse_mode_c
from fusion_diarize.types import AsrStatus, Source, Turn


def _assert_no_cross_system_overlap(turns: list[Turn]) -> None:
    moss = [t for t in turns if t.source in (Source.MOSS, Source.FUSED)]
    diarizen = [t for t in turns if t.source == Source.DIARIZEN]
    for mt in moss:
        for dt in diarizen:
            assert min(mt.end, dt.end) <= max(mt.start, dt.start)


def test_mode_c_normal_returns_stitched_moss_unchanged_without_dedupe():
    diarizen = [Turn(0.0, 10.0, "speaker_0")]
    moss_stitched = [
        Turn(0.0, 5.0, "global:alice", text="hello", source=Source.MOSS),
        Turn(0.1, 5.1, "global:alice", text="again", source=Source.MOSS),
    ]
    meta = [{"ok": True, "incomplete": False, "start": 0.0, "end": 10.0}]

    fused, info = fuse_mode_c(diarizen, moss_stitched, meta)

    assert info["fusion_path"] == "moss_primary"
    assert [(t.start, t.end, t.speaker_id, t.text) for t in fused] == [
        (0.0, 5.0, "global:alice", "hello"),
        (0.1, 5.1, "global:alice", "again"),
    ]
    assert all(t.source == Source.FUSED for t in fused)
    assert all(t is not original for t, original in zip(fused, moss_stitched))


def test_mode_c_all_failed_without_moss_uses_full_diarizen_copy():
    diarizen = [
        Turn(
            0.0,
            10.0,
            "speaker_0",
            text="dz",
            asr_status=AsrStatus.FINAL,
            source=Source.FUSED,
            confidence=0.7,
        )
    ]
    meta = [
        {"ok": False, "start": 0.0, "end": 5.0},
        {"ok": False, "start": 5.0, "end": 10.0},
    ]

    fused, info = fuse_mode_c(diarizen, [], meta)

    assert info["fusion_path"] == "diarizen_backbone_all_moss_failed"
    assert len(fused) == 1
    assert fused[0] is not diarizen[0]
    assert fused[0].source == Source.DIARIZEN
    assert (fused[0].text, fused[0].asr_status, fused[0].confidence) == (
        "dz",
        AsrStatus.FINAL,
        0.7,
    )


def test_mode_c_all_failed_omits_nonpositive_and_tiny_diarizen_turns():
    diarizen = [
        Turn(0.0, 0.0, "zero"),
        Turn(2.0, 1.0, "negative"),
        Turn(3.0, 3.001, "threshold"),
        Turn(4.0, 4.002, "valid"),
    ]
    meta = [{"ok": False, "start": 0.0, "end": 5.0}]

    fused, info = fuse_mode_c(diarizen, [], meta)

    assert info["fusion_path"] == "diarizen_backbone_all_moss_failed"
    assert [(t.start, t.end, t.speaker_id) for t in fused] == [
        (4.0, 4.002, "valid")
    ]


def test_mode_c_span_local_explosion_replaces_only_exploded_interval():
    diarizen = [Turn(0.0, 10.0, "dz")]
    moss_stitched = [
        Turn(0.0, 5.0, "moss_a", text="before", source=Source.MOSS),
        Turn(5.0, 10.0, "moss_b", text="exploded", source=Source.MOSS),
    ]
    meta = [{"ok": True, "start": 0.0, "end": 10.0}]

    fused, info = fuse_mode_c(
        diarizen, moss_stitched, meta, exploded_spans=[(5.0, 10.0)]
    )

    assert info == {
        "fusion_path": "moss_primary_with_explosion_fallback",
        "exploded_spans": [{"start": 5.0, "end": 10.0}],
    }
    assert [(t.start, t.end, t.speaker_id, t.source) for t in fused] == [
        (0.0, 5.0, "moss_a", Source.FUSED),
        (5.0, 10.0, "dz", Source.DIARIZEN),
    ]
    _assert_no_cross_system_overlap(fused)


def test_mode_c_split_moss_text_is_assigned_to_one_longest_fragment():
    diarizen = [Turn(0.0, 10.0, "dz")]
    moss_stitched = [
        Turn(
            0.0,
            10.0,
            "moss",
            text="say this once",
            asr_status=AsrStatus.FINAL,
            source=Source.MOSS,
        )
    ]
    meta = [{"ok": True, "start": 0.0, "end": 10.0}]

    fused, _ = fuse_mode_c(
        diarizen, moss_stitched, meta, exploded_spans=[(4.0, 6.0)]
    )

    moss_fragments = [t for t in fused if t.speaker_id == "moss"]
    assert [
        (t.start, t.end, t.text, t.asr_status, t.source) for t in moss_fragments
    ] == [
        (0.0, 4.0, "say this once", AsrStatus.FINAL, Source.FUSED),
        (6.0, 10.0, "", AsrStatus.EMPTY, Source.MOSS),
    ]


def test_mode_c_clipped_diarizen_fallback_drops_text_and_status():
    diarizen = [
        Turn(
            0.0,
            5.0,
            "dz",
            text="must not duplicate",
            asr_status=AsrStatus.FINAL,
        )
    ]
    meta = [{"ok": True, "start": 0.0, "end": 5.0}]

    fused, _ = fuse_mode_c(
        diarizen,
        [],
        meta,
        exploded_spans=[(3.0, 4.0), (1.0, 2.0)],
    )

    assert [(t.start, t.end, t.text, t.asr_status) for t in fused] == [
        (1.0, 2.0, "", AsrStatus.EMPTY),
        (3.0, 4.0, "", AsrStatus.EMPTY),
    ]


def test_mode_c_explosion_precedes_incomplete_and_gapfills_remainder():
    diarizen = [Turn(0.0, 12.0, "dz")]
    moss_stitched = [
        Turn(0.0, 5.0, "moss_a", text="a", source=Source.MOSS),
        Turn(6.0, 10.0, "moss_b", text="b", source=Source.MOSS),
    ]
    meta = [{"ok": True, "incomplete": True, "start": 4.0, "end": 12.0}]

    fused, info = fuse_mode_c(
        diarizen, moss_stitched, meta, exploded_spans=[(4.0, 6.0)]
    )

    assert info["fusion_path"] == "moss_primary_with_explosion_fallback"
    assert info["exploded_spans"] == [{"start": 4.0, "end": 6.0}]
    assert info["incomplete_spans"] == [{"start": 4.0, "end": 12.0}]
    assert [(t.start, t.end, t.source) for t in fused] == [
        (0.0, 4.0, Source.FUSED),
        (4.0, 6.0, Source.DIARIZEN),
        (6.0, 10.0, Source.FUSED),
        (10.0, 12.0, Source.DIARIZEN),
    ]
    _assert_no_cross_system_overlap(fused)


def test_mode_c_incomplete_gapfill_is_system_exclusive():
    diarizen = [
        Turn(0.0, 10.0, "dz_a"),
        Turn(0.0, 10.0, "dz_b"),
    ]
    moss_stitched = [
        Turn(2.0, 7.0, "moss", text="speech", source=Source.MOSS)
    ]
    meta = [{"ok": False, "start": 0.0, "end": 10.0}]

    fused, info = fuse_mode_c(diarizen, moss_stitched, meta)

    assert info["fusion_path"] == "moss_primary_gapfill"
    assert info["incomplete_spans"] == [{"start": 0.0, "end": 10.0}]
    assert [(t.start, t.end, t.speaker_id, t.source) for t in fused] == [
        (0.0, 2.0, "dz_a", Source.DIARIZEN),
        (0.0, 2.0, "dz_b", Source.DIARIZEN),
        (2.0, 7.0, "moss", Source.FUSED),
        (7.0, 10.0, "dz_a", Source.DIARIZEN),
        (7.0, 10.0, "dz_b", Source.DIARIZEN),
    ]
    _assert_no_cross_system_overlap(fused)


def test_mode_c_retains_true_moss_multitalk_during_gapfill():
    diarizen = [Turn(0.0, 8.0, "dz")]
    moss_stitched = [
        Turn(2.0, 6.0, "alice", text="a", source=Source.MOSS),
        Turn(3.0, 5.0, "bob", text="b", source=Source.MOSS),
    ]
    meta = [{"ok": True, "incomplete": True, "start": 0.0, "end": 8.0}]

    fused, info = fuse_mode_c(diarizen, moss_stitched, meta)

    assert info["fusion_path"] == "moss_primary_gapfill"
    assert {(t.start, t.end, t.speaker_id) for t in fused if t.source == Source.FUSED} == {
        (2.0, 6.0, "alice"),
        (3.0, 5.0, "bob"),
    }
    assert [(t.start, t.end) for t in fused if t.source == Source.DIARIZEN] == [
        (0.0, 2.0),
        (6.0, 8.0),
    ]
