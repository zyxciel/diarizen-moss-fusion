"""Offline pipeline test with FakeDiariZen + FakeMoss (no GPU / HF)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import fusion_diarize.pipeline as pipeline_module
from fusion_diarize.audio_prep import write_mono16k_wav
from fusion_diarize.chunk_planner import (
    DEFAULT_HARD_CAP,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_MAX,
    DEFAULT_TARGET_MIN,
)
from fusion_diarize.diarizen_runner import FakeDiariZenRunner
from fusion_diarize.export import read_json, write_json
from fusion_diarize.identity_stitcher import StitchConfig
from fusion_diarize.moss_runner import FakeMossRunner
from fusion_diarize.pipeline import CHUNK_PLAN_VERSION, run_pipeline
from fusion_diarize.types import AsrStatus, DiarResult, Source, Turn


class CountingDiariZenRunner(FakeDiariZenRunner):
    def __init__(self, turns, centroids, repo_id="fake-diarizen"):
        super().__init__(turns, centroids)
        self.repo_id = repo_id
        self.embed_calls = 0
        self.run_calls = 0

    def run(self, path, sess_name="utt"):
        self.run_calls += 1
        return super().run(path, sess_name)

    def embed_moss_labels(self, path, moss_turns):
        self.embed_calls += 1
        return {
            label: np.array(
                [1.0, 0.0] if label.endswith("S01") else [0.0, 1.0],
                dtype=np.float32,
            )
            for label in {turn.speaker_id for turn in moss_turns}
        }


class CountingMossRunner(FakeMossRunner):
    def __init__(
        self,
        turns_by_default,
        model_path="fake-moss",
        max_new_tokens=1024,
    ):
        super().__init__(turns_by_default)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.run_calls = 0

    def run_chunks(self, full_audio, chunks, work_dir):
        self.run_calls += 1
        return super().run_chunks(full_audio, chunks, work_dir)


def _inputs(tmp_path: Path):
    audio = tmp_path / "utt.wav"
    if not audio.exists():
        write_mono16k_wav(audio, np.zeros(16000, dtype=np.float32))  # 1 s
    diarizen_turns = [
        Turn(0.0, 0.6, "speaker_0"),
        Turn(0.4, 1.0, "speaker_1"),
    ]
    centroids = {
        "speaker_0": np.array([1.0, 0.0], dtype=np.float32),
        "speaker_1": np.array([0.0, 1.0], dtype=np.float32),
    }
    moss_turns = [
        Turn(
            0.05,
            0.55,
            "c000:S01",
            text="hello",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
        Turn(
            0.45,
            0.95,
            "c000:S02",
            text="world",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
    ]
    return audio, diarizen_turns, centroids, moss_turns


def _run(
    tmp_path: Path,
    *,
    mode="c",
    identity_map="hierarchical",
    stitch_config=StitchConfig(),
    force_rechunk=False,
    diarizen_repo="fake-diarizen",
    moss_model="fake-moss",
    max_new_tokens=1024,
    **chunk_parameters,
):
    audio, diarizen_turns, centroids, moss_turns = _inputs(tmp_path)
    diarizen = CountingDiariZenRunner(
        diarizen_turns, centroids, repo_id=diarizen_repo
    )
    moss = CountingMossRunner(
        moss_turns,
        model_path=moss_model,
        max_new_tokens=max_new_tokens,
    )
    outs = run_pipeline(
        audio=audio,
        work_dir=tmp_path / "work",
        mode=mode,
        diarizen_runner=diarizen,
        moss_runner=moss,
        identity_map=identity_map,
        stitch_config=stitch_config,
        force_rechunk=force_rechunk,
        **chunk_parameters,
    )
    return outs, diarizen, moss


def test_run_pipeline_mode_both_writes_exports(tmp_path: Path):
    outs, _, _ = _run(tmp_path, mode="both")
    assert outs["mode_a.json"].is_file()
    assert outs["mode_b.json"].is_file()
    assert outs["mode_c.json"].is_file()
    assert outs["mode_a.rttm"].is_file()
    assert outs["mode_b.rttm"].is_file()
    assert outs["mode_c.rttm"].is_file()

    a = read_json(outs["mode_a.json"])
    b = read_json(outs["mode_b.json"])
    c = read_json(outs["mode_c.json"])
    assert a.meta.get("mode") == "a"
    assert b.meta.get("mode") == "b"
    assert c.meta.get("mode") == "c"
    assert "confidences" in a.meta and "confidences" in b.meta
    assert c.meta["identity_strategy"] == "hierarchical"
    assert "identity_stitching" in c.meta
    assert c.meta["mapping"] == {
        "c000:S01": "speaker_0",
        "c000:S02": "speaker_1",
    }
    assert isinstance(a.turns, list)
    assert isinstance(b.turns, list)
    assert isinstance(c.turns, list)

    work_dir = tmp_path / "work"
    for name in ("diarizen.json", "chunks.json", "moss_turns.json", "prepared.wav", "identity_stitching.json"):
        assert (work_dir / name).is_file()


def test_hierarchical_mode_c_cache_hit_skips_embedding(tmp_path: Path):
    first, first_diarizen, _ = _run(tmp_path)
    assert first_diarizen.embed_calls == 1
    cached = read_json(tmp_path / "work" / "identity_stitching.json")
    assert cached.meta["version"] == "hierarchical_v1"
    assert cached.meta["config"] == vars(StitchConfig())
    assert len(cached.meta["fingerprint"]) == 64
    assert isinstance(cached.meta["exploded_spans"], list)
    first_snapshot = json.loads(first["mode_c.json"].read_text())

    second, second_diarizen, second_moss = _run(tmp_path)
    assert second_diarizen.embed_calls == 0
    assert second_moss.run_calls == 0
    second_snapshot = json.loads(second["mode_c.json"].read_text())
    assert second_snapshot["meta"]["identity_strategy"] == "hierarchical"
    assert first_snapshot == second_snapshot
    assert second_diarizen.run_calls == 0

    manifest = json.loads(
        (tmp_path / "work" / "cache_manifest.json").read_text()
    )
    assert manifest["source_audio"]["path"] == str(
        (tmp_path / "utt.wav").resolve()
    )
    assert manifest["diarizen_model"] == {"repo_id": "fake-diarizen"}
    assert manifest["moss_model"] == {
        "model_path": "fake-moss",
        "max_new_tokens": 1024,
    }
    assert "identity_map" not in json.dumps(manifest)


def test_source_audio_change_invalidates_all_caches(tmp_path: Path):
    _run(tmp_path)
    write_mono16k_wav(
        tmp_path / "utt.wav",
        np.zeros(32000, dtype=np.float32),
    )

    _, diarizen, moss = _run(tmp_path)

    assert diarizen.run_calls == 1
    assert moss.run_calls == 1
    assert diarizen.embed_calls == 1


def test_diarizen_model_change_invalidates_downstream_caches(tmp_path: Path):
    _run(tmp_path)

    _, diarizen, moss = _run(tmp_path, diarizen_repo="other-diarizen")

    assert diarizen.run_calls == 1
    assert moss.run_calls == 1
    assert diarizen.embed_calls == 1


@pytest.mark.parametrize(
    ("overrides", "expected_model"),
    [
        ({"moss_model": "other-moss"}, "other-moss"),
        ({"max_new_tokens": 2048}, "fake-moss"),
    ],
)
def test_moss_configuration_change_invalidates_only_moss_downstream(
    tmp_path: Path, overrides, expected_model
):
    _run(tmp_path)

    _, diarizen, moss = _run(tmp_path, **overrides)

    assert diarizen.run_calls == 0
    assert moss.run_calls == 1
    assert diarizen.embed_calls == 1
    manifest = json.loads(
        (tmp_path / "work" / "cache_manifest.json").read_text()
    )
    assert manifest["moss_model"]["model_path"] == expected_model


def test_chunk_parameter_change_invalidates_chunks_and_downstream(tmp_path: Path):
    _run(tmp_path)

    _, diarizen, moss = _run(
        tmp_path,
        target_min=0.3,
        target_max=0.7,
        hard_cap=0.7,
        overlap=0.1,
    )

    assert diarizen.run_calls == 0
    assert moss.run_calls == 1
    assert diarizen.embed_calls == 1


@pytest.mark.parametrize("manifest_state", ["missing", "invalid"])
def test_missing_or_invalid_manifest_recomputes_every_cache(
    tmp_path: Path, manifest_state: str
):
    _run(tmp_path)
    manifest_path = tmp_path / "work" / "cache_manifest.json"
    if manifest_state == "missing":
        manifest_path.unlink()
    else:
        manifest_path.write_text("[]")

    _, diarizen, moss = _run(tmp_path)

    assert diarizen.run_calls == 1
    assert moss.run_calls == 1
    assert diarizen.embed_calls == 1
    assert isinstance(json.loads(manifest_path.read_text()), dict)


def test_failed_manifest_atomic_replace_preserves_previous_manifest(
    tmp_path: Path, monkeypatch
):
    _run(tmp_path)
    work_dir = tmp_path / "work"
    manifest_path = work_dir / "cache_manifest.json"
    previous = manifest_path.read_bytes()
    real_replace = os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination) == manifest_path:
            raise OSError("replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="replace failed"):
        _run(tmp_path)

    assert manifest_path.read_bytes() == previous
    assert not [
        path for path in work_dir.iterdir() if path.name.endswith(".tmp")
    ]


@pytest.mark.parametrize(
    "cache_name",
    ["prepared.wav", "diarizen.json", "moss_turns.json"],
)
def test_atomic_cache_publication_failure_leaves_no_partial_file(
    tmp_path: Path, monkeypatch, cache_name: str
):
    _run(tmp_path)
    work_dir = tmp_path / "work"
    cache_path = work_dir / cache_name
    cache_path.unlink()
    real_replace = os.replace

    def fail_cache_replace(source, destination):
        if Path(destination) == cache_path:
            raise OSError("replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_cache_replace)

    with pytest.raises(OSError, match="replace failed"):
        _run(tmp_path)

    assert not cache_path.exists()
    assert not [
        path for path in work_dir.iterdir() if path.name.endswith(".tmp")
    ]


@pytest.mark.parametrize("upstream", ["moss", "diarizen"])
def test_upstream_cache_content_change_invalidates_identity(
    tmp_path: Path, upstream: str
):
    _run(tmp_path)
    identity_path = tmp_path / "work" / "identity_stitching.json"
    old_fingerprint = read_json(identity_path).meta["fingerprint"]
    upstream_path = tmp_path / "work" / f"{upstream}.json"
    if upstream == "moss":
        upstream_path = tmp_path / "work" / "moss_turns.json"
        cached = read_json(upstream_path)
        cached.turns[0].end = 0.5
    else:
        cached = read_json(upstream_path)
        cached.turns[0].end = 0.5
    write_json(cached, upstream_path)

    _, diarizen, moss = _run(tmp_path)

    assert diarizen.embed_calls == 1
    assert moss.run_calls == 0
    assert read_json(identity_path).meta["fingerprint"] != old_fingerprint


@pytest.mark.parametrize("cache_name", ["moss_turns.json", "diarizen.json"])
def test_regenerated_inference_cache_explicitly_invalidates_identity(
    tmp_path: Path, monkeypatch, cache_name: str
):
    _run(tmp_path)
    (tmp_path / "work" / cache_name).unlink()
    stitch_calls = 0
    original_stitch = pipeline_module.stitch_identities

    def counting_stitch(*args, **kwargs):
        nonlocal stitch_calls
        stitch_calls += 1
        return original_stitch(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "stitch_identities", counting_stitch)

    _, diarizen, moss = _run(tmp_path)

    assert diarizen.embed_calls == 1
    assert stitch_calls == 1
    assert moss.run_calls == 1
    assert diarizen.run_calls == (1 if cache_name == "diarizen.json" else 0)


def test_failed_identity_atomic_replace_preserves_previous_cache(
    tmp_path: Path, monkeypatch
):
    _run(tmp_path)
    work_dir = tmp_path / "work"
    identity_path = work_dir / "identity_stitching.json"
    previous = identity_path.read_bytes()
    real_replace = os.replace

    def fail_identity_replace(source, destination):
        if Path(destination) == identity_path:
            raise OSError("replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_identity_replace)

    with pytest.raises(OSError, match="replace failed"):
        _run(
            tmp_path,
            stitch_config=StitchConfig(anchor_min_overlap=0.1),
        )

    assert identity_path.read_bytes() == previous
    assert not [
        path for path in work_dir.iterdir() if path.name.endswith(".tmp")
    ]


def test_failed_chunk_atomic_replace_keeps_old_plan_after_dependency_cleanup(
    tmp_path: Path, monkeypatch
):
    _run(tmp_path)
    work_dir = tmp_path / "work"
    chunks_path = work_dir / "chunks.json"
    previous = chunks_path.read_bytes()
    real_replace = os.replace

    def fail_chunk_replace(source, destination):
        if Path(destination) == chunks_path:
            raise OSError("replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_chunk_replace)

    with pytest.raises(OSError, match="replace failed"):
        _run(tmp_path, force_rechunk=True)

    assert chunks_path.read_bytes() == previous
    assert not (work_dir / "moss_turns.json").exists()
    assert not (work_dir / "identity_stitching.json").exists()
    assert not [
        path for path in work_dir.iterdir() if path.name.endswith(".tmp")
    ]


def test_changed_stitch_config_rebuilds_identity_only(tmp_path: Path):
    _run(tmp_path)
    config = StitchConfig(anchor_min_overlap=0.1)
    _, diarizen, moss = _run(tmp_path, stitch_config=config)
    assert diarizen.embed_calls == 1
    assert moss.run_calls == 0
    cache = read_json(tmp_path / "work" / "identity_stitching.json")
    assert cache.meta["config"] == vars(config)


def test_force_rechunk_rebuilds_identity_cache(tmp_path: Path):
    _run(tmp_path)
    cache_path = tmp_path / "work" / "identity_stitching.json"
    original = cache_path.stat().st_mtime_ns
    _, diarizen, moss = _run(tmp_path, force_rechunk=True)
    assert diarizen.embed_calls == 1
    assert moss.run_calls == 1
    assert cache_path.stat().st_mtime_ns >= original


def test_legacy_mode_c_preserves_mapping_confidences(tmp_path: Path):
    outs, diarizen, _ = _run(tmp_path, identity_map="legacy")
    result = read_json(outs["mode_c.json"])
    assert diarizen.embed_calls == 1
    assert result.meta["identity_strategy"] == "legacy"
    assert result.meta["mapping"] == {
        "c000:S01": "speaker_0",
        "c000:S02": "speaker_1",
    }
    assert "confidences" in result.meta
    assert "exploded_spans" not in result.meta


def test_invalid_identity_strategy_rejected_before_work(tmp_path: Path):
    audio, diarizen_turns, centroids, moss_turns = _inputs(tmp_path)
    with pytest.raises(ValueError, match="identity_map"):
        run_pipeline(
            audio,
            tmp_path / "work",
            "c",
            CountingDiariZenRunner(diarizen_turns, centroids),
            CountingMossRunner(moss_turns),
            identity_map="unknown",
        )


@pytest.mark.parametrize(
    "bad_payload",
    [
        [],
        None,
        7,
        "scalar",
        {"meta": [], "turns": [], "centroids": {}},
        {"meta": None, "turns": [], "centroids": {}},
        {"meta": {}, "turns": None, "centroids": {}},
        {"meta": {}, "turns": [None], "centroids": {}},
        {
            "meta": {
                "version": "hierarchical_v1",
                "config": vars(StitchConfig()),
                "identity_stitching": {},
                "mapping": ["not", "a", "mapping"],
                "exploded_spans": [],
            },
            "turns": [],
            "centroids": {},
        },
    ],
    ids=[
        "list-root",
        "null-root",
        "number-root",
        "string-root",
        "list-meta",
        "null-meta",
        "null-turns",
        "null-turn",
        "bad-mapping",
    ],
)
def test_invalid_identity_cache_payload_is_recomputed(
    tmp_path: Path, bad_payload
):
    _run(tmp_path)
    cache_path = tmp_path / "work" / "identity_stitching.json"
    cache_path.write_text(json.dumps(bad_payload))
    _, diarizen, moss = _run(tmp_path)
    assert diarizen.embed_calls == 1
    assert moss.run_calls == 0


def test_identity_cache_version_mismatch_recomputes_identity_only(
    tmp_path: Path, monkeypatch
):
    _run(tmp_path)
    cache_path = tmp_path / "work" / "identity_stitching.json"
    payload = json.loads(cache_path.read_text())
    payload["meta"]["version"] = "hierarchical_v0"
    cache_path.write_text(json.dumps(payload))
    stitch_calls = 0
    original_stitch = pipeline_module.stitch_identities

    def counting_stitch(*args, **kwargs):
        nonlocal stitch_calls
        stitch_calls += 1
        return original_stitch(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "stitch_identities", counting_stitch)

    _, diarizen, moss = _run(tmp_path)

    assert diarizen.embed_calls == 1
    assert stitch_calls == 1
    assert moss.run_calls == 0
    assert read_json(cache_path).meta["version"] == "hierarchical_v1"


def test_legacy_mode_c_falls_back_only_for_exploded_owned_chunk(
    tmp_path: Path,
):
    audio = tmp_path / "utt.wav"
    write_mono16k_wav(audio, np.zeros(32000, dtype=np.float32))
    diarizen_turns = [
        Turn(0.0, 1.0, "speaker_0"),
        Turn(1.0, 2.0, "speaker_1"),
    ]
    centroids = {
        "speaker_0": np.array([1.0, 0.0], dtype=np.float32),
        "speaker_1": np.array([0.0, 1.0], dtype=np.float32),
    }
    moss_turns = [
        Turn(
            0.1,
            0.9,
            "c000:S01",
            source=Source.MOSS,
        ),
        Turn(
            1.1,
            1.4,
            "c001:S02",
            text="exploded one",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
        Turn(
            1.5,
            1.9,
            "c001:S03",
            text="exploded two",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
    ]
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    diarizen = CountingDiariZenRunner(diarizen_turns, centroids)
    moss = CountingMossRunner([])
    write_json(
        DiarResult(
            diarizen_turns,
            meta={"source": "diarizen"},
            centroids={
                key: value.tolist() for key, value in centroids.items()
            },
        ),
        work_dir / "diarizen.json",
    )
    (work_dir / "chunks.json").write_text(
        json.dumps(
            {
                "version": CHUNK_PLAN_VERSION,
                "chunks": [
                    {"start": 0.0, "end": 1.2},
                    {"start": 0.8, "end": 2.0},
                ],
            }
        )
    )
    write_json(
        DiarResult(
            moss_turns,
            meta={
                "moss_chunk_meta": [
                    {"chunk": 0, "start": 0.0, "end": 1.2, "ok": True},
                    {"chunk": 1, "start": 0.8, "end": 2.0, "ok": True},
                ]
            },
        ),
        work_dir / "moss_turns.json",
    )
    (work_dir / "cache_manifest.json").write_text(
        json.dumps(
            pipeline_module._cache_manifest(
                audio,
                diarizen,
                moss,
                target_min=DEFAULT_TARGET_MIN,
                target_max=DEFAULT_TARGET_MAX,
                hard_cap=DEFAULT_HARD_CAP,
                overlap=DEFAULT_OVERLAP,
            ),
            indent=2,
            sort_keys=True,
        )
    )

    outs = run_pipeline(
        audio,
        work_dir,
        "c",
        diarizen,
        moss,
        identity_map="legacy",
        stitch_config=StitchConfig(
            explosion_abs_cap=1,
            explosion_ratio=0.5,
        ),
    )

    result = read_json(outs["mode_c.json"])
    assert moss.run_calls == 0
    assert result.meta["exploded_spans"] == [{"start": 1.0, "end": 2.0}]
    assert any(
        turn.start == 0.1
        and turn.end == 0.9
        and turn.source == Source.MOSS
        for turn in result.turns
    )
    assert not any(
        turn.source in (Source.MOSS, Source.FUSED) and turn.start >= 1.0
        for turn in result.turns
    )
    assert any(
        turn.start == 1.0
        and turn.end == 2.0
        and turn.source == Source.DIARIZEN
        for turn in result.turns
    )
