from __future__ import annotations
from fusion_diarize.types import Turn, ChunkWindow


def _speakers_in_window(turns: list[Turn], start: float, end: float) -> set[str]:
    spk = set()
    for t in turns:
        if t.end > start and t.start < end:
            spk.add(t.speaker_id)
    return spk


def _gap_cut_candidates(turns: list[Turn], win_start: float, search_lo: float, search_hi: float) -> list[float]:
    intervals = sorted((t.start, t.end) for t in turns if t.end > win_start)
    candidates = []
    cursor = win_start
    for s, e in intervals:
        if s > cursor and search_lo <= (cursor + s) / 2 <= search_hi:
            candidates.append((cursor + s) / 2)
        cursor = max(cursor, e)
    return candidates


def plan_chunks(
    turns: list[Turn],
    duration: float,
    target_min: float = 1800.0,
    target_max: float = 3600.0,
    hard_cap: float = 5400.0,
    overlap: float = 45.0,
    max_local_speakers: int = 8,
) -> list[ChunkWindow]:
    if duration <= 0:
        return []
    if duration <= target_max:
        spk = _speakers_in_window(turns, 0.0, duration)
        return [ChunkWindow(0.0, duration, len(spk) > max_local_speakers, len(spk))]

    chunks: list[ChunkWindow] = []
    start = 0.0
    while start < duration - 1e-6:
        remaining = duration - start
        if remaining <= hard_cap:
            end = duration
        else:
            search_lo = start + target_min
            search_hi = start + min(target_max, hard_cap)
            cands = _gap_cut_candidates(turns, start, search_lo, search_hi)
            end = cands[0] if cands else start + min(target_max, hard_cap)
            end = min(end, duration)
        spk = _speakers_in_window(turns, start, end)
        chunks.append(ChunkWindow(start, end, len(spk) > max_local_speakers, len(spk)))
        if end >= duration - 1e-6:
            break
        start = max(0.0, end - overlap)
        if start >= end - 1.0:
            start = end
    return chunks
