from __future__ import annotations
import numpy as np
from fusion_diarize.types import Turn


def _overlap(a: Turn, b: Turn) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def assignment_confidence(overlap: float, cosine: float, overlap_scale: float = 5.0) -> float:
    o = min(1.0, overlap / overlap_scale)
    c = max(0.0, min(1.0, (cosine + 1.0) / 2.0))  # map [-1,1] -> [0,1]
    return 0.5 * o + 0.5 * c


def map_moss_speakers(
    moss_turns: list[Turn],
    diarizen_turns: list[Turn],
    moss_emb: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> dict[str, str]:
    """Greedy: for each local MOSS label, score global DiariZen speakers."""
    local_labels = sorted({t.speaker_id for t in moss_turns})
    global_labels = list(centroids.keys()) or sorted({t.speaker_id for t in diarizen_turns})
    mapping: dict[str, str] = {}
    used_global: set[str] = set()
    scores: list[tuple[float, str, str]] = []
    for loc in local_labels:
        loc_turns = [t for t in moss_turns if t.speaker_id == loc]
        for glob in global_labels:
            glob_turns = [t for t in diarizen_turns if t.speaker_id == glob]
            ov = sum(_overlap(a, b) for a in loc_turns for b in glob_turns)
            cos = 0.0
            if loc in moss_emb and glob in centroids:
                cos = cosine(moss_emb[loc], centroids[glob])
            score = assignment_confidence(ov, cos)
            scores.append((score, loc, glob))
    scores.sort(reverse=True)
    for score, loc, glob in scores:
        if loc in mapping or glob in used_global:
            continue
        mapping[loc] = glob
        used_global.add(glob)
    # leftover locals: best glob even if reused
    for loc in local_labels:
        if loc in mapping:
            continue
        best = None
        best_s = -1.0
        for score, l, glob in scores:
            if l == loc and score > best_s:
                best_s, best = score, glob
        if best is not None:
            mapping[loc] = best
    return mapping


def remap_turns(moss_turns: list[Turn], mapping: dict[str, str]) -> list[Turn]:
    out = []
    for t in moss_turns:
        spk = mapping.get(t.speaker_id, t.speaker_id)
        out.append(Turn(t.start, t.end, spk, t.text, t.asr_status, t.source, t.confidence))
    return out
