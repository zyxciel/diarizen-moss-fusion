"""Orchestrate DiariZen + MOSS fuse with cached work_dir intermediates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fusion_diarize.audio_prep import load_mono16k, probe_duration, write_mono16k_wav
from fusion_diarize.chunk_planner import plan_chunks
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


def _centroids_to_lists(centroids: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in centroids.items()}


def _centroids_from_lists(centroids: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in (centroids or {}).items()
    }


def _save_chunks(path: Path, chunks: list[ChunkWindow]) -> None:
    payload = [
        {
            "start": c.start,
            "end": c.end,
            "high_speaker_density": c.high_speaker_density,
            "n_local_speakers": c.n_local_speakers,
        }
        for c in chunks
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_chunks(path: Path) -> list[ChunkWindow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ChunkWindow(
            float(c["start"]),
            float(c["end"]),
            bool(c.get("high_speaker_density", False)),
            int(c.get("n_local_speakers", 0)),
        )
        for c in payload
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


def run_pipeline(
    audio: Path,
    work_dir: Path,
    mode: str,  # "a" | "b" | "both"
    diarizen_runner,  # duck-typed: .run(path)->(turns, centroids), .embed_moss_labels(path, turns)->dict
    moss_runner,  # .run_chunks(audio, chunks, moss_dir)->(turns, meta)
    tau: float = 0.6,
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
    if chunks_path.is_file():
        chunks = _load_chunks(chunks_path)
    else:
        duration = probe_duration(prepared)
        chunks = plan_chunks(diarizen_turns, duration)
        _save_chunks(chunks_path, chunks)

    moss_dir = work_dir / "moss"
    moss_turns_path = work_dir / "moss_turns.json"
    if moss_turns_path.is_file():
        moss_turns, moss_meta = _load_moss(moss_turns_path)
    else:
        moss_turns, moss_meta = moss_runner.run_chunks(prepared, chunks, moss_dir)
        _save_moss(moss_turns_path, moss_turns, moss_meta)

    moss_emb = diarizen_runner.embed_moss_labels(prepared, moss_turns)
    mapping = map_moss_speakers(moss_turns, diarizen_turns, moss_emb, centroids)
    confidences = _per_label_confidence(
        mapping, moss_turns, diarizen_turns, moss_emb, centroids
    )
    moss_remapped = remap_turns(moss_turns, mapping)

    uri = audio.stem
    outs: dict[str, Path] = {}

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
                },
            ),
            json_b,
        )
        outs["mode_b.rttm"] = rttm_b
        outs["mode_b.json"] = json_b

    return outs
