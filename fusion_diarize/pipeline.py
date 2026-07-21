"""Orchestrate DiariZen + MOSS fuse with cached work_dir intermediates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    write_json(DiarResult(turns=turns, meta={"moss_chunk_meta": meta}), path)


def _load_moss(path: Path) -> tuple[list[Turn], list[dict]]:
    result = read_json(path)
    meta = result.meta.get("moss_chunk_meta", [])
    if not isinstance(meta, list):
        meta = []
    return result.turns, meta


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
    mode: str,  # "a" | "b" | "both"
    diarizen_runner,  # duck-typed: .run(path)->(turns, centroids), .embed_moss_labels(path, turns)->dict
    moss_runner,  # .run_chunks(audio, chunks, moss_dir)->(turns, meta)
    tau: float = 0.6,
    *,
    target_min: float = DEFAULT_TARGET_MIN,
    target_max: float = DEFAULT_TARGET_MAX,
    hard_cap: float = DEFAULT_HARD_CAP,
    overlap: float = DEFAULT_OVERLAP,
    force_rechunk: bool = False,
) -> dict[str, Path]:
    if mode not in ("a", "b", "both"):
        raise ValueError(f"mode must be 'a', 'b', or 'both'; got {mode!r}")

    audio = Path(audio)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared = work_dir / "prepared.wav"
    if not prepared.is_file():
        wav, _sr = load_mono16k(audio)
        write_mono16k_wav(prepared, wav)

    diarizen_path = work_dir / "diarizen.json"
    if diarizen_path.is_file():
        cached = read_json(diarizen_path)
        diarizen_turns = cached.turns
        centroids = _centroids_from_lists(cached.centroids)
    else:
        diarizen_turns, centroids = diarizen_runner.run(prepared)
        write_json(
            DiarResult(
                turns=diarizen_turns,
                meta={"source": "diarizen"},
                centroids=_centroids_to_lists(centroids),
            ),
            diarizen_path,
        )

    chunks_path = work_dir / "chunks.json"
    moss_turns_path = work_dir / "moss_turns.json"
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
        _save_chunks(chunks_path, chunks, version=CHUNK_PLAN_VERSION)
        # Chunk plan changed → invalidate cached MOSS
        if moss_turns_path.is_file():
            moss_turns_path.unlink()

    moss_dir = work_dir / "moss"
    if moss_turns_path.is_file() and not force_rechunk:
        moss_turns, moss_meta = _load_moss(moss_turns_path)
    else:
        moss_turns, moss_meta = moss_runner.run_chunks(prepared, chunks, moss_dir)
        _save_moss(moss_turns_path, moss_turns, moss_meta)

    moss_turns = _filter_moss_on_failed_chunks(moss_turns, moss_meta)

    moss_emb = diarizen_runner.embed_moss_labels(prepared, moss_turns)
    mapping = map_moss_speakers(moss_turns, diarizen_turns, moss_emb, centroids)
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
                confidences[glob] = min(confidences.get(glob, 0.0), tau - 1e-3)

    moss_remapped = remap_turns(moss_turns, mapping)

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

    return outs
