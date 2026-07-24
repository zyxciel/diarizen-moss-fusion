import json
import math
import inspect
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

import fusion_diarize.identity_stitcher as stitcher_module
from fusion_diarize.identity_stitcher import (
    LinkEdge,
    LocalSpeakerNode,
    StitchConfig,
    activity_iou,
    build_embedding_edges,
    build_nodes,
    chunk_ownership_spans,
    cluster_nodes,
    compute_diarizen_anchors,
    detect_exploded_chunks,
    match_adjacent_chunks,
    parse_chunk_index,
    reconcile_chunk_ownership,
    stitch_identities,
)
from fusion_diarize.types import AsrStatus, ChunkWindow, Source, Turn


def turn(start, end, speaker, text="", source=Source.MOSS):
    return Turn(start, end, speaker, text=text, source=source)


def node(local_id, chunk, start, end, embedding=None, anchor=None):
    return LocalSpeakerNode(
        local_id,
        chunk,
        [turn(start, end, local_id)],
        embedding,
        anchor=anchor,
        anchor_purity=1.0 if anchor else 0.0,
    )


def test_parse_chunk_index_requires_namespace():
    assert parse_chunk_index("c12:speaker_3") == 12
    with pytest.raises(ValueError, match="namespaced"):
        parse_chunk_index("speaker_3")


def test_reconcile_keeps_adjacent_textual_turns_distinct():
    reconciled = reconcile_chunk_ownership(
        [
            Turn(
                0,
                1,
                "c0:a",
                text="hello",
                asr_status=AsrStatus.PROVISIONAL,
                source=Source.MOSS,
            ),
            Turn(
                1,
                2,
                "c1:b",
                text="hello",
                asr_status=AsrStatus.PROVISIONAL,
                source=Source.MOSS,
            ),
        ],
        {"c0:a": "speaker_0", "c1:b": "speaker_0"},
        chunk_ownership_spans([ChunkWindow(0, 1), ChunkWindow(1, 2)]),
    )

    assert [(item.start, item.end, item.text) for item in reconciled] == [
        (0, 1, "hello"),
        (1, 2, "hello"),
    ]


def test_reconcile_merges_compatible_empty_activity_conservatively():
    reconciled = reconcile_chunk_ownership(
        [
            Turn(
                0,
                1,
                "c0:a",
                asr_status=AsrStatus.EMPTY,
                source=Source.MOSS,
                confidence=0.9,
            ),
            Turn(
                1,
                2,
                "c1:b",
                asr_status=AsrStatus.EMPTY,
                source=Source.MOSS,
                confidence=0.7,
            ),
        ],
        {"c0:a": "speaker_0", "c1:b": "speaker_0"},
        chunk_ownership_spans([ChunkWindow(0, 1), ChunkWindow(1, 2)]),
    )

    assert len(reconciled) == 1
    assert (reconciled[0].start, reconciled[0].end) == (0, 2)
    assert reconciled[0].text == ""
    assert reconciled[0].asr_status == AsrStatus.EMPTY
    assert reconciled[0].source == Source.MOSS
    assert reconciled[0].confidence == pytest.approx(0.7)


def test_activity_iou_penalizes_false_alarm_activity():
    left = [turn(0, 4, "left")]
    right = [turn(0, 2, "right")]
    iou, intersection, left_s, right_s = activity_iou(left, right, 0, 4)
    assert iou == pytest.approx(0.5)
    assert (intersection, left_s, right_s) == pytest.approx((2, 4, 2))


def test_build_nodes_groups_turns_and_discards_invalid_embeddings():
    turns = [
        turn(0, 1, "c0:a"),
        turn(2, 3, "c0:a"),
        turn(0, 1, "c1:b"),
    ]
    nodes = build_nodes(turns, {"c0:a": [1, 0], "c1:b": [np.nan, 0]})
    assert [n.local_id for n in nodes] == ["c0:a", "c1:b"]
    assert len(nodes[0].turns) == 2
    assert np.array_equal(nodes[0].embedding, np.array([1.0, 0.0]))
    assert nodes[1].embedding is None


def test_adjacent_overlap_links_different_local_ids():
    nodes = [
        node("c0:a", 0, 8, 12, [1, 0]),
        node("c1:z", 1, 8, 12, [1, 0]),
    ]
    chunks = [ChunkWindow(0, 12), ChunkWindow(8, 20)]
    edges, rejected = match_adjacent_chunks(
        nodes,
        chunks,
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert [(e.left, e.right, e.method) for e in edges] == [
        ("c0:a", "c1:z", "moss_overlap")
    ]
    assert edges[0].strength == pytest.approx(1.0)
    assert edges[0].assignment_margin is None
    with pytest.raises(FrozenInstanceError):
        edges[0].strength = 0.0
    json.dumps(rejected, allow_nan=False)


def test_hungarian_rejects_below_threshold_and_ambiguous_margin():
    chunks = [ChunkWindow(0, 4), ChunkWindow(0, 4)]
    below = [
        node("c0:a", 0, 0, 4),
        node("c1:b", 1, 0, 2),
    ]
    edges, rejected = match_adjacent_chunks(
        below,
        chunks,
        StitchConfig(
            min_overlap_activity=0.5,
            min_intersection=0.5,
            overlap_link_score=0.6,
        ),
    )
    assert edges == []
    assert any(item["reason"] == "below_link_score" for item in rejected)

    ambiguous = [
        node("c0:a", 0, 0, 4),
        node("c1:b", 1, 0, 4),
        node("c1:c", 1, 0, 3.8),
    ]
    edges, rejected = match_adjacent_chunks(
        ambiguous,
        chunks,
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert edges == []
    assert any(item["reason"] == "ambiguous_assignment" for item in rejected)
    json.dumps(rejected, allow_nan=False)


def test_hungarian_masks_weak_and_invalid_pairs_before_assignment(monkeypatch):
    scores = {
        ("c0:a", "c1:x"): (0.90, 1.0),
        ("c0:a", "c1:y"): (0.69, 1.0),
        ("c0:b", "c1:x"): (0.69, 1.0),
        ("c0:b", "c1:y"): (0.0, 0.0),
    }

    def fake_activity_iou(left, right, *_args):
        score, intersection = scores[(left[0].speaker_id, right[0].speaker_id)]
        return score, intersection, 1.0, 1.0

    monkeypatch.setattr(stitcher_module, "activity_iou", fake_activity_iou)
    edges, _ = match_adjacent_chunks(
        [
            node("c0:a", 0, 0, 1),
            node("c0:b", 0, 0, 1),
            node("c1:x", 1, 0, 1),
            node("c1:y", 1, 0, 1),
        ],
        [ChunkWindow(0, 1), ChunkWindow(0, 1)],
        StitchConfig(
            min_overlap_activity=0.5,
            min_intersection=0.5,
            overlap_link_score=0.70,
            assignment_margin=0.10,
        ),
    )
    assert [(edge.left, edge.right, edge.strength) for edge in edges] == [
        ("c0:a", "c1:x", pytest.approx(0.90))
    ]


def test_accepted_hungarian_edge_reports_finite_margin():
    edges, _ = match_adjacent_chunks(
        [
            node("c0:a", 0, 0, 4),
            node("c1:b", 1, 0, 4),
            node("c1:c", 1, 0, 2),
        ],
        [ChunkWindow(0, 4), ChunkWindow(0, 4)],
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert len(edges) == 1
    assert edges[0].assignment_margin == pytest.approx(0.5)


def test_strong_overlap_can_override_conflicting_diarizen_anchors():
    nodes = [
        node("c0:a", 0, 0, 4, [1, 0], "dz1"),
        node("c1:b", 1, 0, 4, [1, 0], "dz2"),
    ]
    edges, _ = match_adjacent_chunks(
        nodes,
        [ChunkWindow(0, 4), ChunkWindow(0, 4)],
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert len(edges) == 1


def test_anchor_uses_overlap_purity_duration_and_optional_embedding():
    nodes = [
        node("c0:a", 0, 0, 5, [1, 0]),
        node("c0:b", 0, 0, 5, None),
        node("c0:c", 0, 0, 2, [1, 0]),
    ]
    dz = [
        turn(0, 4, "dz1", source=Source.DIARIZEN),
        turn(4, 5, "dz2", source=Source.DIARIZEN),
    ]
    anchored = compute_diarizen_anchors(
        nodes, dz, {"dz1": [1, 0], "dz2": [-1, 0]}, StitchConfig()
    )
    assert anchored[0].anchor == "dz1"
    assert anchored[0].anchor_purity == pytest.approx(0.8)
    assert anchored[1].anchor == "dz1"  # no embedding does not reject temporal anchor
    assert anchored[2].anchor is None  # insufficient overlap duration


def test_anchor_purity_denominator_sums_all_diarizen_overlap():
    target = node("c0:a", 0, 0, 4)
    anchored = compute_diarizen_anchors(
        [target],
        [
            turn(0, 4, "dz1", source=Source.DIARIZEN),
            turn(0, 2, "dz2", source=Source.DIARIZEN),
        ],
        {},
        StitchConfig(anchor_min_overlap=1.0),
    )
    assert anchored[0].anchor is None  # 4 / (4 + 2) is below .70


def test_anchor_rejects_low_raw_centroid_cosine():
    target = node("c0:a", 0, 0, 4, [1, 0])
    anchored = compute_diarizen_anchors(
        [target],
        [turn(0, 4, "dz1", source=Source.DIARIZEN)],
        {"dz1": [-1, 0]},
        StitchConfig(anchor_min_overlap=1.0),
    )
    assert anchored[0].anchor is None


def test_nonadjacent_embedding_link_and_ambiguous_or_weak_stay_separate():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0]),
        node("c2:a2", 2, 4, 5, [0.99, 0.01]),
    ]
    edges, rejected = build_embedding_edges(nodes, StitchConfig())
    pairs = {frozenset((e.left, e.right)) for e in edges}
    assert frozenset(("c0:a", "c2:a2")) in pairs
    assert edges[0].assignment_margin is not None
    assert math.isfinite(edges[0].assignment_margin)
    _, diagnostics = cluster_nodes(nodes, edges, StitchConfig())
    assert math.isfinite(diagnostics["accepted_edges"][0]["assignment_margin"])

    ambiguous = [
        node("c0:x", 0, 0, 1, [1, 0]),
        node("c1:near", 1, 2, 3, [0.79, 0.613]),
        node("c2:y", 2, 4, 5, [0.99, 0.01]),
        node("c3:z", 3, 6, 7, [0.98, 0.02]),
    ]
    edges, rejected = build_embedding_edges(ambiguous, StitchConfig())
    pairs = {frozenset((e.left, e.right)) for e in edges}
    assert all("c1:near" not in pair for pair in pairs)
    assert not (
        frozenset(("c0:x", "c3:z")) in pairs
        and frozenset(("c2:y", "c3:z")) in pairs
    )
    json.dumps(rejected)


def test_embedding_margin_lookup_does_not_rescan_all_pairs_per_edge():
    source = inspect.getsource(build_embedding_edges)
    assert "for pair, other_cosine in similarities.items()" not in source


def test_embedding_edges_exclude_adjacent_chunks():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0]),
        node("c1:b", 1, 2, 3, [1, 0]),
        node("c2:c", 2, 4, 5, [1, 0]),
    ]
    edges, rejected = build_embedding_edges(nodes, StitchConfig())
    assert all(abs(parse_chunk_index(edge.left) - parse_chunk_index(edge.right)) > 1 for edge in edges)
    assert any(item["reason"] == "adjacent_chunks_use_overlap" for item in rejected)


def test_embedding_edges_respect_conflicting_anchors_and_shared_anchor_is_not_auto_merge():
    conflicting = [
        node("c0:a", 0, 0, 1, [1, 0], "dz1"),
        node("c2:b", 2, 2, 3, [0.82, 0.572], "dz2"),
    ]
    edges, rejected = build_embedding_edges(conflicting, StitchConfig())
    assert edges == []
    assert any(r["reason"] == "conflicting_anchors" for r in rejected)

    shared = [
        node("c0:x", 0, 0, 1, [1, 0], "dz1"),
        node("c2:y", 2, 2, 3, [0, 1], "dz1"),
    ]
    edges, _ = build_embedding_edges(shared, StitchConfig())
    assert edges == []


def test_compatible_anchors_break_embedding_near_tie():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0], "dz1"),
        node("c2:compatible", 2, 2, 3, [0.9, 0.436], "dz1"),
        node("c3:other", 3, 4, 5, [0.89, 0.456], "dz2"),
    ]
    edges, _ = build_embedding_edges(nodes, StitchConfig())
    pairs = {frozenset((edge.left, edge.right)) for edge in edges}
    assert pairs == {frozenset(("c0:a", "c2:compatible"))}


def test_compatible_anchor_does_not_waive_margin_beyond_point_zero_three():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0], "dz1"),
        node("c2:top", 2, 2, 3, [0.9, math.sqrt(1 - 0.9**2)], "dz1"),
        node("c3:runner_up", 3, 4, 5, [0.86, math.sqrt(1 - 0.86**2)], "dz1"),
    ]
    edges, rejected = build_embedding_edges(nodes, StitchConfig())
    assert edges == []
    assert any(
        item["left"] == "c0:a"
        and item["right"] == "c2:top"
        and item["reason"] == "ambiguous_embedding"
        for item in rejected
    )


def test_cluster_enforces_same_chunk_cannot_link():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0]),
        node("c0:b", 0, 0, 1, [1, 0]),
        node("c1:x", 1, 2, 3, [1, 0]),
    ]
    edges = [
        LinkEdge("c0:a", "c1:x", 0.99, "test", embedding_cosine=1),
        LinkEdge("c0:b", "c1:x", 0.98, "test", embedding_cosine=1),
    ]
    mapping, metadata = cluster_nodes(nodes, edges, StitchConfig())
    assert mapping["c0:a"] != mapping["c0:b"]
    assert any(r["reason"] == "same_chunk_cannot_link" for r in metadata["rejected_edges"])


def test_cluster_rejects_abc_chaining_that_exceeds_diameter():
    nodes = [
        node("c0:a", 0, 0, 1, [1, 0]),
        node("c1:b", 1, 2, 3, [0.9, 0.436]),
        node("c2:c", 2, 4, 5, [0.6, 0.8]),
    ]
    edges = [
        LinkEdge("c0:a", "c1:b", 0.95, "test"),
        LinkEdge("c1:b", "c2:c", 0.94, "test"),
    ]
    mapping, metadata = cluster_nodes(nodes, edges, StitchConfig())
    assert mapping["c0:a"] == mapping["c1:b"]
    assert mapping["c0:a"] != mapping["c2:c"]
    rejection = next(
        r for r in metadata["rejected_edges"] if r["reason"] == "cluster_max_distance"
    )
    assert rejection["left"] == "c1:b"
    assert rejection["right"] == "c2:c"
    for diagnostic in metadata["clusters"]:
        assert "maximum_pairwise_distance" in diagnostic
        assert "maximum_medoid_distance" in diagnostic
    merged = next(item for item in metadata["clusters"] if len(item["local_ids"]) == 2)
    assert merged["maximum_pairwise_distance"] == pytest.approx(0.1, abs=1e-3)
    assert merged["maximum_medoid_distance"] == pytest.approx(0.1, abs=1e-3)


def test_final_ids_are_ordered_by_earliest_activity_then_local_id():
    nodes = [
        node("c2:z", 2, 5, 6),
        node("c0:b", 0, 1, 2),
        node("c1:a", 1, 1, 2),
    ]
    mapping, _ = cluster_nodes(nodes, [], StitchConfig())
    assert mapping == {
        "c0:b": "speaker_0",
        "c1:a": "speaker_1",
        "c2:z": "speaker_2",
    }


def test_ownership_preserves_text_boundaries_and_multitalk():
    chunks = [ChunkWindow(0, 10), ChunkWindow(8, 18)]
    spans = chunk_ownership_spans(chunks)
    assert spans == [
        {"chunk_index": 0, "start": 0.0, "end": 9.0},
        {"chunk_index": 1, "start": 9.0, "end": 18.0},
    ]
    raw = [
        turn(8, 10, "c0:a", "left"),
        turn(8, 10, "c1:x", "right"),
        turn(8.5, 9.5, "c1:y", "other"),
    ]
    out = reconcile_chunk_ownership(
        raw,
        {"c0:a": "speaker_0", "c1:x": "speaker_0", "c1:y": "speaker_1"},
        spans,
    )
    speaker0 = [t for t in out if t.speaker_id == "speaker_0"]
    assert [(item.start, item.end, item.text) for item in speaker0] == [
        (8, 9.0, "left"),
        (9.0, 10, "right"),
    ]
    assert any(
        t.speaker_id == "speaker_1" and (t.start, t.end) == (9, 9.5) for t in out
    )


def test_explosion_is_per_chunk_not_global_sum():
    chunks = [ChunkWindow(0, 10), ChunkWindow(8, 18)]
    spans = chunk_ownership_spans(chunks)
    raw = [
        turn(0, 1, f"c0:s{i}") for i in range(7)
    ] + [
        turn(10, 11, f"c1:s{i}") for i in range(7)
    ]
    assert detect_exploded_chunks(raw, [], chunks, spans, StitchConfig()) == []

    exploded_raw = [turn(0, 1, f"c0:s{i}") for i in range(13)]
    exploded = detect_exploded_chunks(
        exploded_raw,
        [turn(0, 1, "dz", source=Source.DIARIZEN)],
        [chunks[0]],
        [spans[0]],
        StitchConfig(),
    )
    assert exploded[0]["chunk_index"] == 0
    assert exploded[0]["n_moss"] == 13
    assert exploded[0]["owned_span"] == {"start": 0.0, "end": 9.0}


def test_missing_embeddings_and_diarizen_are_supported():
    moss = [turn(0, 2, "c0:a")]
    result = stitch_identities(
        moss, [ChunkWindow(0, 2)], [], {}, {}, StitchConfig()
    )
    assert result.mapping == {"c0:a": "speaker_0"}
    assert result.turns[0].speaker_id == "speaker_0"


def test_end_to_end_result_has_json_safe_diagnostics_and_switch_count():
    moss = [
        turn(0, 6, "c0:a", "first"),
        turn(4, 10, "c1:b", "second"),
    ]
    chunks = [ChunkWindow(0, 6), ChunkWindow(4, 10)]
    dz = [turn(0, 10, "dz1", source=Source.DIARIZEN)]
    result = stitch_identities(
        moss,
        chunks,
        dz,
        {"c0:a": [1, 0], "c1:b": [1, 0]},
        {"dz1": [1, 0]},
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert result.mapping["c0:a"] == result.mapping["c1:b"]
    assert result.metadata["version"] == "hierarchical_v1"
    assert result.metadata["assignment_method"] == "scipy_linear_sum_assignment"
    assert result.metadata["cross_chunk_identity_switch_count"] == 0
    assert result.metadata["ownership_spans"]
    assignment = result.metadata["assignments"]["c0:a"]
    assert assignment == {
        "global_id": result.mapping["c0:a"],
        "method": "moss_overlap",
        "confidence": pytest.approx(1.0),
        "diarizen_anchor": "dz1",
        "diarizen_purity": pytest.approx(1.0),
    }
    assert result.metadata["edges"]["accepted"]
    assert result.exploded_spans == []
    json.dumps(result.metadata, allow_nan=False)


def test_exploded_spans_are_tuples_while_details_stay_in_metadata():
    moss = [turn(0, 1, f"c0:s{index}") for index in range(13)]
    result = stitch_identities(
        moss,
        [ChunkWindow(0, 2)],
        [],
        {},
        {},
        StitchConfig(),
    )
    assert result.exploded_spans == [(0.0, 2.0)]
    assert result.metadata["exploded_chunks"][0]["n_moss"] == 13
    json.dumps(result.metadata, allow_nan=False)


def test_switch_count_includes_adjacent_candidate_rejected_by_clustering():
    result = stitch_identities(
        [turn(0, 4, "c0:a"), turn(0, 4, "c1:b")],
        [ChunkWindow(0, 4), ChunkWindow(0, 4)],
        [],
        {"c0:a": [1, 0], "c1:b": [-1, 0]},
        {},
        StitchConfig(min_overlap_activity=0.5, min_intersection=0.5),
    )
    assert result.mapping["c0:a"] != result.mapping["c1:b"]
    assert result.metadata["cross_chunk_identity_switch_count"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_hop": 0.0},
        {"frame_hop": math.nan},
        {"embedding_margin": math.inf},
        {"min_intersection": -0.1},
        {"assignment_margin": -0.1},
        {"overlap_link_score": 1.01},
        {"embedding_cosine": -1.01},
        {"cluster_max_distance": 2.01},
        {"anchor_purity": math.nan},
        {"anchor_min_cosine": 1.01},
        {"explosion_abs_cap": -1},
        {"explosion_ratio": 0.0},
    ],
)
def test_stitch_config_rejects_invalid_numeric_values(kwargs):
    with pytest.raises(ValueError):
        StitchConfig(**kwargs)


def test_default_stitch_config_is_strictly_json_safe():
    json.dumps(asdict(StitchConfig()), allow_nan=False)


def test_stitch_config_normalizes_numpy_scalars_for_strict_json():
    defaults = asdict(StitchConfig())
    config = StitchConfig(
        **{
            name: np.int64(value)
            if name == "explosion_abs_cap"
            else np.float32(value)
            for name, value in defaults.items()
        }
    )
    assert type(config.explosion_abs_cap) is int
    assert all(
        type(getattr(config, name)) is float
        for name in defaults
        if name != "explosion_abs_cap"
    )
    json.dumps(asdict(config), allow_nan=False)
