"""Mode A fusion: MOSS local refine + DiariZen gap fill (no whole-track replace)."""
from __future__ import annotations

from fusion_diarize.types import Turn, AsrStatus, Source


def _clip_interval(start: float, end: float, cut_start: float, cut_end: float) -> list[tuple[float, float]]:
    """Return remnants of [start, end] after removing [cut_start, cut_end]."""
    if end <= cut_start or start >= cut_end or end <= start:
        return [(start, end)] if end > start else []
    out: list[tuple[float, float]] = []
    if start < cut_start:
        out.append((start, min(end, cut_start)))
    if end > cut_end:
        out.append((max(start, cut_end), end))
    return [(a, b) for a, b in out if b - a > 1e-3]


def subtract_coverage(
    turn: Turn, covered: list[tuple[float, float]]
) -> list[Turn]:
    """Split ``turn`` around covered intervals; drop near-empty remnants."""
    pieces = [(turn.start, turn.end)]
    for cs, ce in covered:
        next_pieces: list[tuple[float, float]] = []
        for a, b in pieces:
            next_pieces.extend(_clip_interval(a, b, cs, ce))
        pieces = next_pieces
    return [
        Turn(
            a,
            b,
            turn.speaker_id,
            "",
            AsrStatus.EMPTY,
            Source.DIARIZEN,
            turn.confidence,
        )
        for a, b in pieces
        if b - a > 1e-3
    ]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + 1e-3:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def dedupe_overlapping_turns(turns: list[Turn], iou_thresh: float = 0.5) -> list[Turn]:
    """Drop near-duplicate turns (e.g. chunk-overlap MOSS copies). Keep earlier."""
    kept: list[Turn] = []
    for t in sorted(turns, key=lambda x: (x.start, x.end)):
        drop = False
        for k in kept:
            if k.speaker_id != t.speaker_id:
                continue
            inter = max(0.0, min(k.end, t.end) - max(k.start, t.start))
            union = max(k.end, t.end) - min(k.start, t.start)
            if union > 0 and inter / union >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(t)
    return kept


def fuse_mode_a(
    diarizen: list[Turn],
    moss_remapped: list[Turn],
    confidences: dict[str, float],
    tau: float = 0.6,
) -> list[Turn]:
    """Segment-level Mode A fusion.

    Previous v1 dropped *all* DiariZen turns for a high-conf speaker and kept
    every MOSS turn for that speaker. That caused:
    - **high FA**: chunk-overlap MOSS duplicates, and unmapped MOSS locals kept
      alongside DiariZen for the same speech;
    - **high miss**: incomplete MOSS (truncation) left gaps after DiariZen was
      fully removed for that speaker.

    New behavior:
    1. Keep high-conf MOSS turns (after overlap dedupe).
    2. Keep DiariZen only in regions *not* covered by those MOSS turns
       (same speaker), filling MOSS gaps.
    3. Do not keep unmapped / low-conf MOSS that still overlaps DiariZen speech.
    """
    high = {s for s, c in confidences.items() if c >= tau}
    diarizen_spk = {t.speaker_id for t in diarizen}

    moss_trusted = [
        t for t in moss_remapped if t.speaker_id in high and t.speaker_id in diarizen_spk
    ]
    moss_trusted = dedupe_overlapping_turns(moss_trusted)

    out: list[Turn] = []
    for t in moss_trusted:
        out.append(
            Turn(
                t.start,
                t.end,
                t.speaker_id,
                t.text,
                AsrStatus.PROVISIONAL if t.text else AsrStatus.EMPTY,
                Source.FUSED,
                confidences.get(t.speaker_id, 1.0),
            )
        )

    # Per-speaker MOSS coverage used to carve DiariZen gaps.
    coverage: dict[str, list[tuple[float, float]]] = {}
    for t in moss_trusted:
        coverage.setdefault(t.speaker_id, []).append((t.start, t.end))
    for spk, ivals in list(coverage.items()):
        coverage[spk] = _merge_intervals(ivals)

    for t in diarizen:
        covered = coverage.get(t.speaker_id, [])
        if not covered:
            out.append(
                Turn(
                    t.start,
                    t.end,
                    t.speaker_id,
                    "",
                    AsrStatus.EMPTY,
                    Source.DIARIZEN,
                    1.0,
                )
            )
        else:
            out.extend(subtract_coverage(t, covered))

    # Unmapped / low-conf MOSS: keep only if it does not overlap DiariZen speech
    # (avoids dual hypotheses on the same region → FA).
    diarizen_intervals = [(t.start, t.end) for t in diarizen]
    for t in moss_remapped:
        if t.speaker_id in high and t.speaker_id in diarizen_spk:
            continue
        if t.speaker_id in diarizen_spk and confidences.get(t.speaker_id, 0.0) < tau:
            continue  # low-conf mapped: trust DiariZen
        # Unmapped local id (e.g. c000:S03): keep only if little DiariZen overlap
        ov = sum(
            max(0.0, min(t.end, b) - max(t.start, a)) for a, b in diarizen_intervals
        )
        if ov / max(t.end - t.start, 1e-6) > 0.3:
            continue
        out.append(
            Turn(
                t.start,
                t.end,
                t.speaker_id,
                t.text,
                AsrStatus.PROVISIONAL if t.text else AsrStatus.EMPTY,
                Source.MOSS,
                confidences.get(t.speaker_id, 0.3),
            )
        )

    out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
    return out
