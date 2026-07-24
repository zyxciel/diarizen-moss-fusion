"""Mode C fusion: stitched MOSS primary with span-local DiariZen fallback."""
from __future__ import annotations

from typing import Any

from fusion_diarize.fuse_a import _merge_intervals
from fusion_diarize.types import AsrStatus, Source, Turn

_MIN_DURATION = 1e-3


def _copy_turn(
    turn: Turn,
    *,
    start: float | None = None,
    end: float | None = None,
    source: Source | None = None,
) -> Turn:
    return Turn(
        turn.start if start is None else start,
        turn.end if end is None else end,
        turn.speaker_id,
        turn.text,
        turn.asr_status,
        turn.source if source is None else source,
        turn.confidence,
    )


def _valid_spans(
    spans: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    return _merge_intervals(
        [
            (float(start), float(end))
            for start, end in spans or []
            if end - start > _MIN_DURATION
        ]
    )


def _subtract_spans(turn: Turn, spans: list[tuple[float, float]]) -> list[Turn]:
    """Subtract sorted, merged spans from a turn in one linear sweep."""
    intervals: list[tuple[float, float]] = []
    cursor = turn.start
    for cut_start, cut_end in spans:
        if cut_end <= cursor:
            continue
        if cut_start >= turn.end:
            break
        piece_end = min(cut_start, turn.end)
        if piece_end - cursor > _MIN_DURATION:
            intervals.append((cursor, piece_end))
        cursor = max(cursor, cut_end)
        if cursor >= turn.end:
            break
    if turn.end - cursor > _MIN_DURATION:
        intervals.append((cursor, turn.end))
    return [
        _copy_turn(turn, start=start, end=end)
        for start, end in intervals
    ]


def _intersect_spans(turn: Turn, spans: list[tuple[float, float]]) -> list[Turn]:
    return [
        _copy_turn(turn, start=max(turn.start, start), end=min(turn.end, end))
        for start, end in spans
        if min(turn.end, end) - max(turn.start, start) > _MIN_DURATION
    ]


def _moss_source(turn: Turn) -> Source:
    return Source.FUSED if turn.text else Source.MOSS


def _moss_fragments(turn: Turn, excluded: list[tuple[float, float]]) -> list[Turn]:
    fragments = _subtract_spans(turn, excluded)
    if turn.text and len(fragments) > 1:
        text_index = max(
            range(len(fragments)),
            key=lambda index: (
                fragments[index].end - fragments[index].start,
                -fragments[index].start,
            ),
        )
        for index, fragment in enumerate(fragments):
            if index != text_index:
                fragment.text = ""
                fragment.asr_status = AsrStatus.EMPTY
    for fragment in fragments:
        fragment.source = _moss_source(fragment)
    return fragments


def _as_diarizen_fallback(turn: Turn) -> Turn:
    turn.text = ""
    turn.asr_status = AsrStatus.EMPTY
    turn.source = Source.DIARIZEN
    return turn


def _incomplete_spans(moss_meta: list[dict]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for chunk in moss_meta:
        if not (chunk.get("incomplete") or not chunk.get("ok", True)):
            continue
        if "start" not in chunk or "end" not in chunk:
            continue
        spans.append((float(chunk["start"]), float(chunk["end"])))
    return _valid_spans(spans)


def _json_spans(spans: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"start": start, "end": end} for start, end in spans]


def fuse_mode_c(
    diarizen: list[Turn],
    moss_stitched: list[Turn],
    moss_meta: list[dict],
    *,
    exploded_spans: list[tuple[float, float]] | None = None,
) -> tuple[list[Turn], dict[str, Any]]:
    """Fuse globally stitched MOSS with span-local DiariZen fallback."""
    if (
        not moss_stitched
        and moss_meta
        and all(chunk.get("ok") is False for chunk in moss_meta)
    ):
        out = [
            _copy_turn(turn, source=Source.DIARIZEN)
            for turn in diarizen
            if turn.end - turn.start > _MIN_DURATION
        ]
        out.sort(key=lambda turn: (turn.start, turn.end, turn.speaker_id))
        return out, {"fusion_path": "diarizen_backbone_all_moss_failed"}

    exploded = _valid_spans(exploded_spans)
    incomplete = _incomplete_spans(moss_meta)

    out: list[Turn] = []
    for turn in moss_stitched:
        out.extend(_moss_fragments(turn, exploded))

    for turn in diarizen:
        for piece in _intersect_spans(turn, exploded):
            out.append(_as_diarizen_fallback(piece))

    moss_activity = _merge_intervals(
        [
            (turn.start, turn.end)
            for turn in out
            if turn.source in (Source.MOSS, Source.FUSED)
        ]
    )
    unavailable_for_gapfill = _merge_intervals(moss_activity + exploded)
    if incomplete:
        for turn in diarizen:
            for candidate in _intersect_spans(turn, incomplete):
                for piece in _subtract_spans(candidate, unavailable_for_gapfill):
                    out.append(_as_diarizen_fallback(piece))

    out.sort(key=lambda turn: (turn.start, turn.end, turn.speaker_id))

    if exploded:
        meta: dict[str, Any] = {
            "fusion_path": "moss_primary_with_explosion_fallback",
            "exploded_spans": _json_spans(exploded),
        }
        if incomplete:
            meta["incomplete_spans"] = _json_spans(incomplete)
        return out, meta
    if incomplete:
        return out, {
            "fusion_path": "moss_primary_gapfill",
            "incomplete_spans": _json_spans(incomplete),
        }
    return out, {"fusion_path": "moss_primary"}
