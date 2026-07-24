# Hierarchical Speaker Identity Stitching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Mode C's DiariZen-first greedy speaker mapper with overlap-first hierarchical MOSS identity stitching, constrained embedding clustering, per-chunk failure guards, and duplicate-free overlap ownership.

**Architecture:** A new `identity_stitcher` module builds one node per namespaced MOSS chunk speaker, links adjacent nodes using frame-IoU plus WeSpeaker similarity, uses DiariZen only as an anchor/tie-breaker, and rejects merges that violate same-chunk or cluster-diameter constraints. It reconciles overlapping chunks into one owned MOSS timeline before Mode C applies span-local DiariZen fallback.

**Tech Stack:** Python 3.10+, NumPy, SciPy Hungarian assignment, dataclasses, pytest.

**Design:** `docs/superpowers/specs/2026-07-23-hierarchical-identity-stitching-design.md`

**Git constraint:** Do not create commits unless the user explicitly requests them.

---

## File structure

- Create `fusion_diarize/identity_stitcher.py`: node construction, scoring, anchors, constrained clustering, ownership reconciliation, result metadata.
- Modify `fusion_diarize/diarizen_runner.py`: duration-weighted pooling for namespaced MOSS speaker embeddings.
- Modify `fusion_diarize/fuse_c.py`: consume stitched MOSS turns and apply per-span explosion/incomplete/all-failed fallbacks.
- Modify `fusion_diarize/pipeline.py`: strategy selection, identity cache, hierarchical Mode C integration, legacy A/B mapping.
- Modify `fusion_diarize/cli.py`: `--identity-map hierarchical|legacy`.
- Modify `pyproject.toml`: declare SciPy because hierarchical matching requires `linear_sum_assignment`.
- Create `tests/test_identity_stitcher.py`: scoring, links, constraints, chaining, ownership, explosion.
- Modify `tests/test_fuse_c.py`: span-local explosion and all-MOSS-failed behavior.
- Modify `tests/test_pipeline_offline.py`: hierarchical default, cache, metadata, legacy compatibility.
- Modify `tests/test_diarizen_runner_unit.py`: weighted embedding pooling.
- Modify `docs/remote_smoke.md`: invocation and diagnostics.

## Task 1: Identity model, parsing, and activity IoU

**Files:**
- Create: `fusion_diarize/identity_stitcher.py`
- Create: `tests/test_identity_stitcher.py`

- [ ] **Step 1: Write failing tests for local IDs and temporal IoU**

Add tests that establish exact namespace parsing and frame activity behavior:

```python
import numpy as np
import pytest

from fusion_diarize.identity_stitcher import (
    StitchConfig,
    activity_iou,
    parse_chunk_index,
)
from fusion_diarize.types import Turn


def test_parse_chunk_index():
    assert parse_chunk_index("c003:S07") == 3
    with pytest.raises(ValueError, match="namespaced"):
        parse_chunk_index("S07")


def test_activity_iou_penalizes_one_sided_false_alarm():
    left = [Turn(0.0, 4.0, "c000:S01")]
    right = [Turn(0.0, 2.0, "c001:S03")]
    score, intersection, left_active, right_active = activity_iou(
        left, right, 0.0, 4.0, frame_hop=0.02
    )
    assert score == pytest.approx(0.5)
    assert intersection == pytest.approx(2.0)
    assert left_active == pytest.approx(4.0)
    assert right_active == pytest.approx(2.0)


def test_activity_iou_handles_true_multisegment_activity():
    left = [
        Turn(0.0, 1.0, "c000:S01"),
        Turn(2.0, 3.0, "c000:S01"),
    ]
    right = [
        Turn(0.0, 1.0, "c001:S04"),
        Turn(2.0, 3.0, "c001:S04"),
    ]
    assert activity_iou(left, right, 0.0, 3.0)[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: collection fails because `fusion_diarize.identity_stitcher` does not exist.

- [ ] **Step 3: Implement configuration, node/result dataclasses, parsing, and IoU**

Define these public types and defaults:

```python
@dataclass(frozen=True)
class StitchConfig:
    frame_hop: float = 0.02
    min_overlap_activity: float = 2.0
    min_intersection: float = 1.0
    overlap_link_score: float = 0.70
    assignment_margin: float = 0.10
    embedding_cosine: float = 0.80
    embedding_margin: float = 0.05
    conflicting_anchor_cosine: float = 0.85
    cluster_max_distance: float = 0.30
    anchor_purity: float = 0.70
    anchor_min_overlap: float = 3.0
    anchor_min_cosine: float = 0.40
    explosion_abs_cap: int = 12
    explosion_ratio: float = 2.0


@dataclass
class LocalSpeakerNode:
    local_id: str
    chunk_index: int
    turns: list[Turn]
    embedding: np.ndarray | None
    anchor: str | None = None
    anchor_purity: float = 0.0


@dataclass(frozen=True)
class LinkEdge:
    left: str
    right: str
    strength: float
    method: str
    temporal_iou: float | None = None
    embedding_cosine: float | None = None


@dataclass
class StitchResult:
    turns: list[Turn]
    mapping: dict[str, str]
    metadata: dict[str, Any]
    exploded_spans: list[tuple[float, float]]
```

Implement `parse_chunk_index` with the regex `^c(\d+):.+$`. Implement `activity_iou` by clipping turns to the shared window, painting boolean NumPy masks at `frame_hop`, and returning `(iou, intersection_seconds, left_seconds, right_seconds)`. A zero union returns four zeros.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: the three tests pass.

## Task 2: Nodes, DiariZen anchors, and adjacent Hungarian links

**Files:**
- Modify: `fusion_diarize/identity_stitcher.py`
- Modify: `tests/test_identity_stitcher.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing tests for node grouping and adjacent matching**

```python
from fusion_diarize.identity_stitcher import (
    build_nodes,
    compute_diarizen_anchors,
    match_adjacent_chunks,
)
from fusion_diarize.types import ChunkWindow


def test_adjacent_overlap_links_different_local_ids():
    chunks = [ChunkWindow(0.0, 10.0), ChunkWindow(6.0, 16.0)]
    turns = [
        Turn(6.0, 10.0, "c000:S01"),
        Turn(6.0, 10.0, "c001:S03"),
    ]
    embeddings = {
        "c000:S01": np.array([1.0, 0.0]),
        "c001:S03": np.array([0.99, 0.01]),
    }
    nodes = build_nodes(turns, embeddings)
    edges, rejected = match_adjacent_chunks(nodes, chunks, StitchConfig(
        min_overlap_activity=1.0,
        min_intersection=1.0,
    ))
    assert [(e.left, e.right) for e in edges] == [
        ("c000:S01", "c001:S03")
    ]
    assert rejected == []


def test_diarizen_anchor_requires_purity_and_embedding_agreement():
    nodes = build_nodes(
        [Turn(0.0, 8.0, "c000:S01")],
        {"c000:S01": np.array([1.0, 0.0])},
    )
    compute_diarizen_anchors(
        nodes,
        [
            Turn(0.0, 7.0, "speaker_0"),
            Turn(7.0, 8.0, "speaker_1"),
        ],
        {
            "speaker_0": np.array([1.0, 0.0]),
            "speaker_1": np.array([0.0, 1.0]),
        },
        StitchConfig(),
    )
    assert nodes["c000:S01"].anchor == "speaker_0"
    assert nodes["c000:S01"].anchor_purity == pytest.approx(0.875)
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: imports fail for the new functions.

- [ ] **Step 3: Declare SciPy and implement matching**

Change:

```toml
dependencies = ["numpy", "scipy"]
```

Implement:

- `build_nodes(turns, embeddings)` grouped by namespaced local ID;
- `compute_diarizen_anchors(nodes, diarizen_turns, centroids, config)` using overlap purity and raw embedding cosine;
- `match_adjacent_chunks(nodes, chunks, config)` using `scipy.optimize.linear_sum_assignment`.

For each adjacent chunk pair, derive the shared window as:

```python
overlap_start = max(chunks[i].start, chunks[i + 1].start)
overlap_end = min(chunks[i].end, chunks[i + 1].end)
```

Eligibility requires both active durations and intersection duration to pass configuration gates. Score is `0.7 * iou + 0.3 * normalized_cosine`; when either embedding is missing, score is IoU. Reject assigned pairs below score threshold or whose row/column alternative margin is below `assignment_margin`. Return accepted edges plus structured rejection diagnostics.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: all node, anchor, and adjacent-link tests pass.

## Task 3: Non-adjacent candidates and constrained clustering

**Files:**
- Modify: `fusion_diarize/identity_stitcher.py`
- Modify: `tests/test_identity_stitcher.py`

- [ ] **Step 1: Add failing tests for conservative embedding linkage**

```python
from fusion_diarize.identity_stitcher import (
    build_embedding_edges,
    cluster_nodes,
)


def test_nonadjacent_high_cosine_links_speaker():
    nodes = build_nodes(
        [
            Turn(0.0, 3.0, "c000:S01"),
            Turn(30.0, 33.0, "c003:S02"),
        ],
        {
            "c000:S01": np.array([1.0, 0.0]),
            "c003:S02": np.array([0.99, 0.01]),
        },
    )
    edges, _ = build_embedding_edges(nodes, [], StitchConfig())
    mapping, meta = cluster_nodes(nodes, edges, StitchConfig())
    assert mapping["c000:S01"] == mapping["c003:S02"]


def test_same_chunk_speakers_never_merge():
    nodes = build_nodes(
        [
            Turn(0.0, 3.0, "c000:S01"),
            Turn(4.0, 7.0, "c000:S02"),
        ],
        {
            "c000:S01": np.array([1.0, 0.0]),
            "c000:S02": np.array([0.99, 0.01]),
        },
    )
    forced = [LinkEdge("c000:S01", "c000:S02", 0.99, "embedding")]
    mapping, meta = cluster_nodes(nodes, forced, StitchConfig())
    assert mapping["c000:S01"] != mapping["c000:S02"]
    assert meta["rejected_edges"][0]["reason"] == "same_chunk_cannot_link"


def test_chaining_merge_rejected_by_cluster_diameter():
    nodes = build_nodes(
        [
            Turn(0.0, 2.0, "c000:S01"),
            Turn(3.0, 5.0, "c001:S01"),
            Turn(6.0, 8.0, "c002:S01"),
        ],
        {
            "c000:S01": np.array([1.0, 0.0]),
            "c001:S01": np.array([0.9, 0.4359]),
            "c002:S01": np.array([0.6, 0.8]),
        },
    )
    forced = [
        LinkEdge("c000:S01", "c001:S01", 0.9, "overlap"),
        LinkEdge("c001:S01", "c002:S01", 0.8, "overlap"),
    ]
    mapping, meta = cluster_nodes(
        nodes, forced, StitchConfig(cluster_max_distance=0.30)
    )
    assert mapping["c000:S01"] != mapping["c002:S01"]
    assert any(
        r["reason"] == "cluster_diameter"
        for r in meta["rejected_edges"]
    )
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: imports fail for embedding edges and clustering.

- [ ] **Step 3: Implement conservative edges and graph clustering**

Implement raw cosine candidate generation for nodes from different chunks:

- reject below `embedding_cosine`;
- require `embedding_margin` over alternatives;
- if accepted anchors conflict, require `conflicting_anchor_cosine`;
- use compatible anchors to resolve candidates whose cosine differs by at most 0.03;
- never create same-chunk edges.

Implement constrained union-find that processes `LinkEdge`s by descending strength. Before each union:

1. reject if the merged component repeats a chunk index;
2. calculate maximum raw cosine distance `1 - cosine` over all available embedding pairs;
3. reject if the diameter exceeds `cluster_max_distance`.

Run a final connected-component validation over accepted edges; split an invalid component by removing its weakest edge until every component passes. Assign deterministic `speaker_N` IDs by earliest turn start and then local ID. Return mapping and full accepted/rejected edge metadata.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: all constrained-clustering tests pass.

## Task 4: Overlap ownership and per-chunk explosion detection

**Files:**
- Modify: `fusion_diarize/identity_stitcher.py`
- Modify: `tests/test_identity_stitcher.py`

- [ ] **Step 1: Add failing ownership and explosion tests**

```python
from fusion_diarize.identity_stitcher import (
    chunk_ownership_spans,
    detect_exploded_chunks,
    reconcile_chunk_ownership,
)


def test_overlap_midpoint_has_single_chunk_owner_and_keeps_multitalk():
    chunks = [ChunkWindow(0.0, 10.0), ChunkWindow(6.0, 16.0)]
    ownership = chunk_ownership_spans(chunks)
    assert ownership == {0: (0.0, 8.0), 1: (8.0, 16.0)}
    turns = [
        Turn(6.0, 10.0, "c000:S01"),
        Turn(6.0, 10.0, "c001:S03"),
        Turn(8.0, 9.0, "c001:S04"),
    ]
    mapping = {
        "c000:S01": "speaker_0",
        "c001:S03": "speaker_0",
        "c001:S04": "speaker_1",
    }
    out = reconcile_chunk_ownership(turns, chunks, mapping)
    assert not any(
        a.speaker_id == b.speaker_id
        and a is not b
        and min(a.end, b.end) > max(a.start, b.start)
        for a in out for b in out
    )
    assert any(t.speaker_id == "speaker_1" for t in out)


def test_explosion_is_per_chunk_not_sum_of_namespaced_ids():
    chunks = [ChunkWindow(0.0, 10.0), ChunkWindow(8.0, 18.0)]
    moss = [
        Turn(0.0, 1.0, f"c000:S{i:02d}") for i in range(8)
    ] + [
        Turn(8.0, 9.0, f"c001:S{i:02d}") for i in range(8)
    ]
    diarizen = [
        Turn(0.0, 18.0, f"speaker_{i}") for i in range(5)
    ]
    exploded = detect_exploded_chunks(moss, chunks, diarizen, StitchConfig())
    assert exploded == []
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: imports fail for ownership and explosion functions.

- [ ] **Step 3: Implement ownership, clipping, merging, and explosion**

`chunk_ownership_spans` places the boundary at each adjacent overlap midpoint. `reconcile_chunk_ownership`:

1. parses each original local ID to locate its chunk owner span;
2. clips each turn to that span;
3. applies the global mapping;
4. retains text/status/source/confidence;
5. merges overlapping or adjacent same-speaker pieces without combining different speakers.

`detect_exploded_chunks` counts unique local MOSS IDs per chunk and unique DiariZen speakers active in that chunk. It applies both `> explosion_abs_cap` and `> explosion_ratio * max(n_dz, 1)`, then converts exploded chunk indexes to ownership spans.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: ownership and per-chunk explosion tests pass.

## Task 5: Stitcher orchestration and cache payload

**Files:**
- Modify: `fusion_diarize/identity_stitcher.py`
- Modify: `tests/test_identity_stitcher.py`

- [ ] **Step 1: Add an end-to-end stitcher test**

```python
from fusion_diarize.identity_stitcher import stitch_identities


def test_stitch_identities_returns_global_turns_and_diagnostics():
    chunks = [ChunkWindow(0.0, 10.0), ChunkWindow(6.0, 16.0)]
    moss = [
        Turn(1.0, 4.0, "c000:S01"),
        Turn(6.0, 10.0, "c000:S01"),
        Turn(6.0, 10.0, "c001:S03"),
        Turn(12.0, 15.0, "c001:S03"),
    ]
    result = stitch_identities(
        moss,
        chunks,
        [Turn(0.0, 16.0, "speaker_7")],
        {
            "c000:S01": np.array([1.0, 0.0]),
            "c001:S03": np.array([0.99, 0.01]),
        },
        {"speaker_7": np.array([1.0, 0.0])},
        StitchConfig(min_overlap_activity=1.0),
    )
    assert set(result.mapping.values()) == {"speaker_0"}
    assert {t.speaker_id for t in result.turns} == {"speaker_0"}
    assert result.metadata["version"] == "hierarchical_v1"
    assert result.metadata["assignments"]["c001:S03"]["method"] == "moss_overlap"
    assert result.metadata["cross_chunk_identity_switch_count"] == 0
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: `stitch_identities` is missing.

- [ ] **Step 3: Implement the public orchestrator**

Implement this exact public signature:

```python
def stitch_identities(
    moss_turns: list[Turn],
    chunks: list[ChunkWindow],
    diarizen_turns: list[Turn],
    moss_embeddings: dict[str, np.ndarray],
    diarizen_centroids: dict[str, np.ndarray],
    config: StitchConfig = StitchConfig(),
) -> StitchResult:
```

The function must:

1. build nodes and anchors;
2. generate adjacent and non-adjacent edges;
3. run constrained clustering;
4. reconcile chunk ownership;
5. detect exploded owned spans;
6. produce JSON-safe metadata containing version, config (`dataclasses.asdict`), assignments, accepted/rejected edges, cluster diagnostics, ownership spans, exploded chunks, and `cross_chunk_identity_switch_count`.

Ensure metadata contains only lists, dictionaries, strings, numbers, booleans, and nulls; convert NumPy scalar values explicitly.

Define `cross_chunk_identity_switch_count` as the number of rejected adjacent-overlap candidate pairs whose score passes `overlap_link_score` but whose two local IDs finish in different global clusters. This makes conservative split decisions visible during evaluation without requiring reference speaker names.

- [ ] **Step 4: Run the entire stitcher test file**

Run:

```bash
python -m pytest tests/test_identity_stitcher.py -q
```

Expected: all stitcher tests pass.

## Task 6: Duration-weighted MOSS speaker embeddings

**Files:**
- Modify: `fusion_diarize/diarizen_runner.py`
- Modify: `tests/test_diarizen_runner_unit.py`

- [ ] **Step 1: Add a failing pooling test**

Use a runner created without model initialization and monkeypatch `_embed_region` so two crops have unequal durations:

```python
def test_embed_moss_labels_is_duration_weighted(monkeypatch):
    from fusion_diarize.diarizen_runner import DiariZenRunner

    runner = object.__new__(DiariZenRunner)
    values = iter([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])
    monkeypatch.setattr(runner, "_embed_region", lambda wav, start, end: next(values))
    monkeypatch.setattr(
        "fusion_diarize.diarizen_runner.load_mono16k",
        lambda path: (np.zeros(16000 * 4, dtype=np.float32), 16000),
    )
    out = runner.embed_moss_labels(
        Path("unused.wav"),
        [
            Turn(0.0, 1.0, "c000:S01"),
            Turn(1.0, 4.0, "c000:S01"),
        ],
    )
    assert out["c000:S01"] == pytest.approx(np.array([0.25, 0.75]))
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_diarizen_runner_unit.py -q
```

Expected: current equal mean returns `[0.5, 0.5]`.

- [ ] **Step 3: Implement duration-weighted valid-vector pooling**

Store `(embedding, max(turn.end - turn.start, 1e-3))` per label and call `np.average(stack, axis=0, weights=weights)`. Continue skipping zero/non-finite embeddings.

- [ ] **Step 4: Run runner tests**

Run:

```bash
python -m pytest tests/test_diarizen_runner_unit.py -q
```

Expected: all tests pass.

## Task 7: Refactor Mode C to span-local fallbacks

**Files:**
- Modify: `fusion_diarize/fuse_c.py`
- Modify: `tests/test_fuse_c.py`

- [ ] **Step 1: Replace whole-file explosion tests with span-local behavior**

Add/adjust tests:

```python
def test_mode_c_explosion_replaces_only_owned_span():
    moss = [
        Turn(0.0, 5.0, "speaker_0", text="normal"),
        Turn(5.0, 10.0, "speaker_1", text="bad"),
    ]
    diarizen = [
        Turn(0.0, 10.0, "dz_0"),
        Turn(6.0, 9.0, "dz_1"),
    ]
    fused, meta = fuse_mode_c(
        diarizen,
        moss,
        [{"ok": True, "incomplete": False, "start": 0.0, "end": 10.0}],
        exploded_spans=[(5.0, 10.0)],
    )
    assert any(t.text == "normal" and t.end <= 5.0 for t in fused)
    assert not any(t.text == "bad" for t in fused)
    assert any(t.source == Source.DIARIZEN and t.start >= 5.0 for t in fused)
    assert meta["fusion_path"] == "moss_primary_with_explosion_fallback"


def test_mode_c_all_moss_failed_uses_full_diarizen():
    fused, meta = fuse_mode_c(
        [Turn(0.0, 10.0, "speaker_0")],
        [],
        [{"ok": False, "incomplete": True, "start": 0.0, "end": 10.0}],
    )
    assert len(fused) == 1
    assert fused[0].source == Source.DIARIZEN
    assert meta["fusion_path"] == "diarizen_backbone_all_moss_failed"
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_fuse_c.py -q
```

Expected: signature mismatch and old whole-file explosion assertions fail.

- [ ] **Step 3: Simplify Mode C API and implement precedence**

Use:

```python
def fuse_mode_c(
    diarizen: list[Turn],
    moss_stitched: list[Turn],
    moss_meta: list[dict],
    *,
    exploded_spans: list[tuple[float, float]] | None = None,
) -> tuple[list[Turn], dict[str, Any]]:
```

Precedence:

1. no stitched MOSS and all chunks failed → full DiariZen;
2. exploded owned spans → remove MOSS in those spans and insert clipped DiariZen there;
3. incomplete/failed spans not already exploded → keep MOSS and fill only MOSS-silent intervals;
4. normal spans → stitched MOSS unchanged.

Use interval clipping/subtraction helpers so DiariZen and MOSS never coexist solely because they came from different systems. Preserve true overlap inside the selected system.

- [ ] **Step 4: Run Mode C tests**

Run:

```bash
python -m pytest tests/test_fuse_c.py -q
```

Expected: all Mode C tests pass.

## Task 8: Pipeline strategy, identity cache, and CLI

**Files:**
- Modify: `fusion_diarize/pipeline.py`
- Modify: `fusion_diarize/cli.py`
- Modify: `tests/test_pipeline_offline.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Add failing pipeline tests**

Add a concrete shared fixture and assert Mode C hierarchical metadata:

```python
def _offline_inputs(tmp_path):
    audio = tmp_path / "utt.wav"
    write_mono16k_wav(audio, np.zeros(16000, dtype=np.float32))
    diarizen = FakeDiariZenRunner(
        [Turn(0.0, 1.0, "speaker_0")],
        {"speaker_0": np.array([1.0, 0.0], dtype=np.float32)},
    )
    moss = FakeMossRunner([
        Turn(
            0.05,
            0.95,
            "c000:S01",
            text="hello",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        )
    ])
    return audio, tmp_path / "work", diarizen, moss


def test_pipeline_mode_c_defaults_to_hierarchical_and_caches_identity(tmp_path):
    audio, work_dir, diarizen, moss = _offline_inputs(tmp_path)
    outs = run_pipeline(
        audio=audio,
        work_dir=work_dir,
        mode="c",
        identity_map="hierarchical",
        diarizen_runner=diarizen,
        moss_runner=moss,
    )
    result = read_json(outs["mode_c.json"])
    assert result.meta["identity_stitching"]["version"] == "hierarchical_v1"
    assert (work_dir / "identity_stitching.json").is_file()


def test_pipeline_mode_both_keeps_legacy_for_ab_and_hierarchical_for_c(tmp_path):
    audio, work_dir, diarizen, moss = _offline_inputs(tmp_path)
    outs = run_pipeline(
        audio=audio,
        work_dir=work_dir,
        mode="both",
        identity_map="hierarchical",
        diarizen_runner=diarizen,
        moss_runner=moss,
    )
    assert {"mode_a.json", "mode_b.json", "mode_c.json"} <= set(outs)
    assert read_json(outs["mode_c.json"]).meta["identity_strategy"] == "hierarchical"
```

Extract `build_parser()` from `main()` and add:

```python
from fusion_diarize.cli import build_parser


def test_cli_identity_map_default_and_override():
    parser = build_parser()
    common = [
        "run",
        "--audio", "a.wav",
        "--work-dir", "work",
        "--moss-model", "moss",
    ]
    assert parser.parse_args(common).identity_map == "hierarchical"
    assert parser.parse_args(common + ["--identity-map", "legacy"]).identity_map == "legacy"
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/test_pipeline_offline.py -q
```

Expected: `run_pipeline` does not accept `identity_map` and no identity cache exists.

- [ ] **Step 3: Add versioned cache helpers and strategy branch**

In `pipeline.py` define:

```python
IDENTITY_STITCH_VERSION = "hierarchical_v1"
```

Save `identity_stitching.json` as a `DiarResult` whose turns are stitched turns and whose meta contains `identity_stitching`, exploded spans, and a cache key derived from version plus `StitchConfig`. Load only when both version and config match.

Branch behavior:

- compute legacy mapping when Mode A/B is requested or `identity_map == "legacy"`;
- compute/load hierarchical stitching for Mode C when `identity_map == "hierarchical"`;
- Mode C legacy remains available for ablation but marks `identity_strategy="legacy"`;
- invalid strategy raises `ValueError`.

Ensure rechunking removes `identity_stitching.json` as well as MOSS cache, because local IDs and ownership spans change.

- [ ] **Step 4: Wire CLI**

Pass through:

```python
run.add_argument(
    "--identity-map",
    choices=["hierarchical", "legacy"],
    default="hierarchical",
    help="Mode C cross-chunk speaker identity strategy",
)
```

Refactor parser creation into `build_parser()` for direct unit testing; `main()` calls `build_parser()` and otherwise remains unchanged.

- [ ] **Step 5: Run integration tests**

Run:

```bash
python -m pytest tests/test_pipeline_offline.py tests/test_fuse_c.py -q
```

Expected: all pass.

## Task 9: Documentation and complete verification

**Files:**
- Modify: `docs/remote_smoke.md`
- Test: all `tests/`

- [ ] **Step 1: Document invocation, cache, and diagnostics**

Update the remote command to show the default:

```bash
fusion-diarize run \
  --audio /data/utt.wav \
  --work-dir /data/work/utt \
  --mode c \
  --identity-map hierarchical \
  --moss-model /models/MOSS-Transcribe-Diarize
```

Document:

- `identity_stitching.json`;
- `mode_c.json → meta.identity_stitching`;
- legacy ablation using `--identity-map legacy`;
- deleting `identity_stitching.json` when manually changing identity thresholds in development;
- per-chunk `exploded_chunks` and rejected edge diagnostics.
- `cross_chunk_identity_switch_count` as the direct cross-chunk fragmentation diagnostic.

- [ ] **Step 2: Run formatting/static checks available in the repository**

Run:

```bash
python -m compileall -q fusion_diarize
git diff --check
```

Expected: both exit zero.

- [ ] **Step 3: Run the full unit suite**

Run:

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify CLI help**

Run:

```bash
python -m fusion_diarize.cli run --help
```

Expected: help lists `--identity-map {hierarchical,legacy}` and Mode C remains the default mode.

- [ ] **Step 5: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: only planned implementation, tests, plan, and documentation are modified; no generated audio/cache artifacts are tracked.
