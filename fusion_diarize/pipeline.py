"""Orchestrate DiariZen + MOSS fuse with cached work_dir intermediates."""
from __future__ import annotations

import json
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import numpy as np

from fusion_diarize.audio_prep import load_mono16k, probe_duration, write_mono16k_wav
from fusion_diarize.chunk_planner import (
    DEFAULT_HARD_CAP,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_MAX,
    DEFAULT_TARGET_MIN,
    plan_chunks,
)
from fusion_diarize.export import read_json, write_json, write_rttm
from fusion_diarize.fuse_a import fuse_mode_a
from fusion_diarize.fuse_b import fuse_mode_b
from fusion_diarize.fuse_c import fuse_mode_c
from fusion_diarize.identity_stitcher import (
    StitchConfig,
    StitchResult,
    chunk_ownership_spans,
    detect_exploded_chunks,
    stitch_identities,
)
from fusion_diarize.mapper import (
    _overlap,
    assignment_confidence,
    cosine,
    map_moss_speakers,
    remap_turns,
)
from fusion_diarize.types import ChunkWindow, DiarResult, Turn

# Bump when chunk defaults change so stale work_dir caches are rebuilt.
CHUNK_PLAN_VERSION = "20min_v1"
IDENTITY_CACHE_VERSION = "hierarchical_v1"
PIPELINE_CACHE_VERSION = "pipeline_v1"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_diar_result(result: DiarResult, path: Path) -> None:
    payload = {
        "meta": result.meta,
        "centroids": result.centroids,
        "turns": [turn.to_dict() for turn in result.turns],
    }
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
    )


def _atomic_publish_file(path: Path, writer: Callable[[Path], None]) -> None:
    """Atomically publish ``path`` via a same-directory temp file.

    The temp suffix keeps the destination extension (e.g. ``.tmp.wav``) so
    format-sensitive writers like ``torchaudio.save`` still recognize the
    container. A bare ``.tmp`` suffix raises ``Unsupported format: tmp``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp{path.suffix}" if path.suffix else ".tmp"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _runner_class_identity(runner: Any) -> str:
    runner_type = type(runner)
    return f"{runner_type.__module__}.{runner_type.__qualname__}"


def _cache_manifest(
    audio: Path,
    diarizen_runner: Any,
    moss_runner: Any,
    *,
    target_min: float,
    target_max: float,
    hard_cap: float,
    overlap: float,
) -> dict[str, Any]:
    stat = audio.stat()
    repo_id = getattr(diarizen_runner, "repo_id", None)
    model_path = getattr(moss_runner, "model_path", None)
    max_new_tokens = getattr(moss_runner, "max_new_tokens", None)
    return {
        "pipeline_cache_version": PIPELINE_CACHE_VERSION,
        "source_audio": {
            "path": str(audio.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "diarizen_model": (
            {"repo_id": str(repo_id)}
            if repo_id is not None
            else {"runner_class": _runner_class_identity(diarizen_runner)}
        ),
        "moss_model": {
            "model_path": (
                str(model_path)
                if model_path is not None
                else f"class:{_runner_class_identity(moss_runner)}"
            ),
            "max_new_tokens": (
                int(max_new_tokens) if max_new_tokens is not None else None
            ),
        },
        "chunk_parameters": {
            "target_min": float(target_min),
            "target_max": float(target_max),
            "hard_cap": float(hard_cap),
            "overlap": float(overlap),
        },
    }


def _load_cache_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    required = {
        "pipeline_cache_version",
        "source_audio",
        "diarizen_model",
        "moss_model",
        "chunk_parameters",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return None
    if payload.get("pipeline_cache_version") != PIPELINE_CACHE_VERSION:
        return None
    if not all(
        isinstance(payload.get(key), dict)
        for key in (
            "source_audio",
            "diarizen_model",
            "moss_model",
            "chunk_parameters",
        )
    ):
        return None
    return payload


def _manifest_invalidations(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> set[str]:
    all_caches = {"prepared", "diarizen", "chunks", "moss", "identity"}
    if previous is None:
        return all_caches
    invalidated: set[str] = set()
    if previous["source_audio"] != current["source_audio"]:
        invalidated.update(all_caches)
    if previous["diarizen_model"] != current["diarizen_model"]:
        invalidated.update({"diarizen", "chunks", "moss", "identity"})
    if previous["chunk_parameters"] != current["chunk_parameters"]:
        invalidated.update({"chunks", "moss", "identity"})
    if previous["moss_model"] != current["moss_model"]:
        invalidated.update({"moss", "identity"})
    return invalidated


def _invalidate_cache(
    name: str,
    *,
    prepared: Path,
    diarizen: Path,
    chunks: Path,
    moss_turns: Path,
    moss_dir: Path,
    identity: Path,
) -> None:
    paths = {
        "prepared": prepared,
        "diarizen": diarizen,
        "chunks": chunks,
        "identity": identity,
    }
    if name == "moss":
        moss_turns.unlink(missing_ok=True)
        if moss_dir.is_dir():
            shutil.rmtree(moss_dir)
        return
    paths[name].unlink(missing_ok=True)


def _centroids_to_lists(centroids: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in centroids.items()}


def _centroids_from_lists(centroids: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in (centroids or {}).items()
    }


def _save_chunks(path: Path, chunks: list[ChunkWindow], *, version: str) -> None:
    payload = {
        "version": version,
        "chunks": [
            {
                "start": c.start,
                "end": c.end,
                "high_speaker_density": c.high_speaker_density,
                "n_local_speakers": c.n_local_speakers,
            }
            for c in chunks
        ],
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False))


def _load_chunks(path: Path, *, expected_version: str) -> list[ChunkWindow] | None:
    """Return chunks if cache version matches; else None (force replan)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Legacy list format (pre-version) → invalidate
    if isinstance(payload, list):
        return None
    if payload.get("version") != expected_version:
        return None
    return [
        ChunkWindow(
            float(c["start"]),
            float(c["end"]),
            bool(c.get("high_speaker_density", False)),
            int(c.get("n_local_speakers", 0)),
        )
        for c in payload.get("chunks", [])
    ]


def _save_moss(path: Path, turns: list[Turn], meta: list[dict]) -> None:
    _atomic_write_diar_result(
        DiarResult(turns=turns, meta={"moss_chunk_meta": meta}),
        path,
    )


def _load_moss(path: Path) -> tuple[list[Turn], list[dict]]:
    result = read_json(path)
    meta = result.meta.get("moss_chunk_meta", [])
    if not isinstance(meta, list):
        meta = []
    return result.turns, meta


def _save_identity_cache(
    path: Path,
    result: StitchResult,
    config: StitchConfig,
    fingerprint: str,
) -> None:
    _atomic_write_diar_result(
        DiarResult(
            turns=result.turns,
            meta={
                "version": IDENTITY_CACHE_VERSION,
                "config": asdict(config),
                "fingerprint": fingerprint,
                "identity_stitching": result.metadata,
                "mapping": result.mapping,
                "exploded_spans": [
                    [float(start), float(end)]
                    for start, end in result.exploded_spans
                ],
            },
        ),
        path,
    )


def _load_identity_cache(
    path: Path, config: StitchConfig, fingerprint: str
) -> StitchResult | None:
    try:
        cached = read_json(path)
        meta = cached.meta
        if not isinstance(meta, dict):
            return None
        if meta.get("version") != IDENTITY_CACHE_VERSION:
            return None
        if meta.get("config") != asdict(config):
            return None
        if meta.get("fingerprint") != fingerprint:
            return None
        stitching = meta.get("identity_stitching")
        mapping = meta.get("mapping")
        spans = meta.get("exploded_spans")
        if not isinstance(stitching, dict):
            return None
        if not isinstance(mapping, dict) or not all(
            isinstance(local, str) and isinstance(global_id, str)
            for local, global_id in mapping.items()
        ):
            return None
        if not isinstance(spans, list):
            return None
        exploded_spans: list[tuple[float, float]] = []
        for span in spans:
            if (
                not isinstance(span, list)
                or len(span) != 2
                or isinstance(span[0], bool)
                or isinstance(span[1], bool)
                or not isinstance(span[0], (int, float))
                or not isinstance(span[1], (int, float))
            ):
                return None
            start, end = float(span[0]), float(span[1])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                return None
            exploded_spans.append((start, end))
        if not isinstance(cached.turns, list) or not all(
            isinstance(turn, Turn) for turn in cached.turns
        ):
            return None
        return StitchResult(cached.turns, mapping, stitching, exploded_spans)
    except (
        AttributeError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        return None


def _identity_fingerprint(
    chunks: list[ChunkWindow],
    moss_turns: list[Turn],
    moss_meta: list[dict],
    diarizen_turns: list[Turn],
    centroids: dict[str, np.ndarray],
    config: StitchConfig,
) -> str:
    payload = {
        "version": IDENTITY_CACHE_VERSION,
        "config": asdict(config),
        "chunks": [
            {
                "start": float(chunk.start),
                "end": float(chunk.end),
                "high_speaker_density": bool(chunk.high_speaker_density),
                "n_local_speakers": int(chunk.n_local_speakers),
            }
            for chunk in chunks
        ],
        "moss_turns": [turn.to_dict() for turn in moss_turns],
        "moss_meta": moss_meta,
        "diarizen_turns": [turn.to_dict() for turn in diarizen_turns],
        "centroids": {
            speaker: np.asarray(value).tolist()
            for speaker, value in centroids.items()
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _per_label_confidence(
    mapping: dict[str, str],
    moss_turns: list[Turn],
    diarizen_turns: list[Turn],
    moss_emb: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> dict[str, float]:
    """Confidence keyed by mapped global speaker id (max if many locals share)."""
    conf: dict[str, float] = {}
    for loc, glob in mapping.items():
        loc_turns = [t for t in moss_turns if t.speaker_id == loc]
        glob_turns = [t for t in diarizen_turns if t.speaker_id == glob]
        ov = sum(_overlap(a, b) for a in loc_turns for b in glob_turns)
        cos = 0.0
        if loc in moss_emb and glob in centroids:
            cos = cosine(moss_emb[loc], centroids[glob])
        score = assignment_confidence(ov, cos)
        conf[glob] = max(conf.get(glob, 0.0), score)
    return conf


def _filter_moss_on_failed_chunks(
    moss_turns: list[Turn], moss_meta: list[dict]
) -> list[Turn]:
    """Drop MOSS turns from failed chunks; keep incomplete turns (DiariZen fills gaps)."""
    failed_spans = [
        (float(m["start"]), float(m["end"]))
        for m in moss_meta
        if not m.get("ok", True)
    ]
    if not failed_spans:
        return moss_turns
    kept: list[Turn] = []
    for t in moss_turns:
        mid = 0.5 * (t.start + t.end)
        if any(a <= mid <= b for a, b in failed_spans):
            continue
        kept.append(t)
    return kept


def run_pipeline(
    audio: Path,
    work_dir: Path,
    mode: str,  # "a" | "b" | "c" | "both"
    diarizen_runner,  # duck-typed: .run(path)->(turns, centroids), .embed_moss_labels(path, turns)->dict
    moss_runner,  # .run_chunks(audio, chunks, moss_dir)->(turns, meta)
    tau: float = 0.6,
    *,
    target_min: float = DEFAULT_TARGET_MIN,
    target_max: float = DEFAULT_TARGET_MAX,
    hard_cap: float = DEFAULT_HARD_CAP,
    overlap: float = DEFAULT_OVERLAP,
    force_rechunk: bool = False,
    identity_map: str = "hierarchical",
    stitch_config: StitchConfig = StitchConfig(),
) -> dict[str, Path]:
    if mode not in ("a", "b", "c", "both"):
        raise ValueError(f"mode must be 'a', 'b', 'c', or 'both'; got {mode!r}")
    if identity_map not in ("hierarchical", "legacy"):
        raise ValueError(
            "identity_map must be 'hierarchical' or 'legacy'; "
            f"got {identity_map!r}"
        )

    audio = Path(audio)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared = work_dir / "prepared.wav"
    diarizen_path = work_dir / "diarizen.json"
    chunks_path = work_dir / "chunks.json"
    moss_turns_path = work_dir / "moss_turns.json"
    moss_dir = work_dir / "moss"
    identity_path = work_dir / "identity_stitching.json"
    manifest_path = work_dir / "cache_manifest.json"
    current_manifest = _cache_manifest(
        audio,
        diarizen_runner,
        moss_runner,
        target_min=target_min,
        target_max=target_max,
        hard_cap=hard_cap,
        overlap=overlap,
    )
    previous_manifest = _load_cache_manifest(manifest_path)
    for cache_name in _manifest_invalidations(
        previous_manifest, current_manifest
    ):
        _invalidate_cache(
            cache_name,
            prepared=prepared,
            diarizen=diarizen_path,
            chunks=chunks_path,
            moss_turns=moss_turns_path,
            moss_dir=moss_dir,
            identity=identity_path,
        )

    if not prepared.is_file():
        wav, _sr = load_mono16k(audio)
        _atomic_publish_file(
            prepared,
            lambda temporary: write_mono16k_wav(temporary, wav),
        )

    diarizen_regenerated = False
    if diarizen_path.is_file():
        cached = read_json(diarizen_path)
        diarizen_turns = cached.turns
        centroids = _centroids_from_lists(cached.centroids)
    else:
        diarizen_regenerated = True
        for cache_name in ("chunks", "moss", "identity"):
            _invalidate_cache(
                cache_name,
                prepared=prepared,
                diarizen=diarizen_path,
                chunks=chunks_path,
                moss_turns=moss_turns_path,
                moss_dir=moss_dir,
                identity=identity_path,
            )
        diarizen_turns, centroids = diarizen_runner.run(prepared)
        _atomic_write_diar_result(
            DiarResult(
                turns=diarizen_turns,
                meta={"source": "diarizen"},
                centroids=_centroids_to_lists(centroids),
            ),
            diarizen_path,
        )
    if diarizen_regenerated:
        identity_path.unlink(missing_ok=True)

    chunks: list[ChunkWindow] | None = None
    if chunks_path.is_file() and not force_rechunk:
        chunks = _load_chunks(chunks_path, expected_version=CHUNK_PLAN_VERSION)
    if chunks is None:
        duration = probe_duration(prepared)
        chunks = plan_chunks(
            diarizen_turns,
            duration,
            target_min=target_min,
            target_max=target_max,
            hard_cap=hard_cap,
            overlap=overlap,
        )
        # Invalidate dependents before publishing the replacement chunk plan.
        for cache_name in ("moss", "identity"):
            _invalidate_cache(
                cache_name,
                prepared=prepared,
                diarizen=diarizen_path,
                chunks=chunks_path,
                moss_turns=moss_turns_path,
                moss_dir=moss_dir,
                identity=identity_path,
            )
        _save_chunks(chunks_path, chunks, version=CHUNK_PLAN_VERSION)

    moss_regenerated = False
    if moss_turns_path.is_file() and not force_rechunk:
        moss_turns, moss_meta = _load_moss(moss_turns_path)
    else:
        moss_regenerated = True
        moss_turns, moss_meta = moss_runner.run_chunks(prepared, chunks, moss_dir)
        _save_moss(moss_turns_path, moss_turns, moss_meta)
    if moss_regenerated:
        identity_path.unlink(missing_ok=True)

    moss_turns = _filter_moss_on_failed_chunks(moss_turns, moss_meta)

    hierarchical_result: StitchResult | None = None
    hierarchical_needed = mode in ("c", "both") and identity_map == "hierarchical"
    identity_fingerprint: str | None = None
    if hierarchical_needed:
        identity_fingerprint = _identity_fingerprint(
            chunks,
            moss_turns,
            moss_meta,
            diarizen_turns,
            centroids,
            stitch_config,
        )
    if hierarchical_needed and identity_path.is_file() and not force_rechunk:
        assert identity_fingerprint is not None
        hierarchical_result = _load_identity_cache(
            identity_path,
            stitch_config,
            identity_fingerprint,
        )

    legacy_needed = mode in ("a", "b", "both") or (
        mode == "c" and identity_map == "legacy"
    )
    moss_emb: dict[str, np.ndarray] | None = None
    if legacy_needed or (hierarchical_needed and hierarchical_result is None):
        moss_emb = diarizen_runner.embed_moss_labels(prepared, moss_turns)

    mapping: dict[str, str] = {}
    confidences: dict[str, float] = {}
    moss_remapped: list[Turn] = []
    if legacy_needed:
        assert moss_emb is not None
        mapping = map_moss_speakers(
            moss_turns, diarizen_turns, moss_emb, centroids
        )
        confidences = _per_label_confidence(
            mapping, moss_turns, diarizen_turns, moss_emb, centroids
        )
        # Lower confidence for speakers whose MOSS evidence mostly sits in incomplete chunks
        incomplete_spans = [
            (float(m["start"]), float(m["end"]))
            for m in moss_meta
            if m.get("incomplete")
        ]
        if incomplete_spans:
            for loc, glob in mapping.items():
                loc_turns = [t for t in moss_turns if t.speaker_id == loc]
                if not loc_turns:
                    continue
                in_incomplete = 0
                for t in loc_turns:
                    mid = 0.5 * (t.start + t.end)
                    if any(a <= mid <= b for a, b in incomplete_spans):
                        in_incomplete += 1
                if in_incomplete / len(loc_turns) >= 0.5:
                    confidences[glob] = min(
                        confidences.get(glob, 0.0), tau - 1e-3
                    )
        moss_remapped = remap_turns(moss_turns, mapping)

    if hierarchical_needed and hierarchical_result is None:
        assert moss_emb is not None
        assert identity_fingerprint is not None
        hierarchical_result = stitch_identities(
            moss_turns,
            chunks,
            diarizen_turns,
            moss_emb,
            centroids,
            stitch_config,
        )
        _save_identity_cache(
            identity_path,
            hierarchical_result,
            stitch_config,
            identity_fingerprint,
        )

    uri = audio.stem
    outs: dict[str, Path] = {}
    n_incomplete = sum(1 for m in moss_meta if m.get("incomplete"))
    n_failed = sum(1 for m in moss_meta if not m.get("ok", True))

    if mode in ("a", "both"):
        fused_a = fuse_mode_a(diarizen_turns, moss_remapped, confidences, tau=tau)
        rttm_a = work_dir / "mode_a.rttm"
        json_a = work_dir / "mode_a.json"
        write_rttm(fused_a, rttm_a, uri=uri)
        write_json(
            DiarResult(
                turns=fused_a,
                meta={
                    "mode": "a",
                    "tau": tau,
                    "mapping": mapping,
                    "confidences": confidences,
                    "moss_chunk_meta": moss_meta,
                    "chunk_plan_version": CHUNK_PLAN_VERSION,
                    "n_chunks": len(chunks),
                    "n_incomplete_moss_chunks": n_incomplete,
                    "n_failed_moss_chunks": n_failed,
                },
            ),
            json_a,
        )
        outs["mode_a.rttm"] = rttm_a
        outs["mode_a.json"] = json_a

    if mode in ("b", "both"):
        fused_b = fuse_mode_b(diarizen_turns, moss_remapped)
        rttm_b = work_dir / "mode_b.rttm"
        json_b = work_dir / "mode_b.json"
        write_rttm(fused_b, rttm_b, uri=uri)
        write_json(
            DiarResult(
                turns=fused_b,
                meta={
                    "mode": "b",
                    "mapping": mapping,
                    "confidences": confidences,
                    "moss_chunk_meta": moss_meta,
                    "chunk_plan_version": CHUNK_PLAN_VERSION,
                    "n_chunks": len(chunks),
                    "n_incomplete_moss_chunks": n_incomplete,
                    "n_failed_moss_chunks": n_failed,
                },
            ),
            json_b,
        )
        outs["mode_b.rttm"] = rttm_b
        outs["mode_b.json"] = json_b

    if mode in ("c", "both"):
        if identity_map == "hierarchical":
            assert hierarchical_result is not None
            c_turns = hierarchical_result.turns
            c_mapping = hierarchical_result.mapping
            exploded_spans = hierarchical_result.exploded_spans
            identity_meta = {
                "identity_strategy": "hierarchical",
                "identity_stitching": hierarchical_result.metadata,
                "mapping": c_mapping,
            }
        else:
            ownership_spans = chunk_ownership_spans(chunks)
            exploded_chunks = detect_exploded_chunks(
                moss_turns,
                diarizen_turns,
                chunks,
                ownership_spans,
                stitch_config,
            )
            exploded_spans = [
                (
                    float(item["owned_span"]["start"]),
                    float(item["owned_span"]["end"]),
                )
                for item in exploded_chunks
            ]
            c_turns = moss_remapped
            identity_meta = {
                "identity_strategy": "legacy",
                "mapping": mapping,
                "confidences": confidences,
            }
        fused_c, fuse_c_meta = fuse_mode_c(
            diarizen_turns,
            c_turns,
            moss_meta,
            exploded_spans=exploded_spans,
        )
        rttm_c = work_dir / "mode_c.rttm"
        json_c = work_dir / "mode_c.json"
        write_rttm(fused_c, rttm_c, uri=uri)
        write_json(
            DiarResult(
                turns=fused_c,
                meta={
                    "mode": "c",
                    **identity_meta,
                    "moss_chunk_meta": moss_meta,
                    "chunk_plan_version": CHUNK_PLAN_VERSION,
                    "n_chunks": len(chunks),
                    "n_incomplete_moss_chunks": n_incomplete,
                    "n_failed_moss_chunks": n_failed,
                    **fuse_c_meta,
                },
            ),
            json_c,
        )
        outs["mode_c.rttm"] = rttm_c
        outs["mode_c.json"] = json_c

    _atomic_write_text(
        manifest_path,
        json.dumps(
            current_manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
    )
    return outs
