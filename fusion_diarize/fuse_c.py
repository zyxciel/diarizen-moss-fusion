"""Mode C fusion: MOSS-primary with DiariZen gap-fill and explosion guard.

System exclusivity: on any time interval, output comes from **one** system only.
MOSS is always kept; DiariZen may only fill regions with **no** MOSS speech
(and only inside incomplete/failed chunk spans). True multi-speaker overlap
*within* MOSS (or within DiariZen-only gaps) is preserved.
"""
from __future__ import annotations

from typing import Any

from fusion_diarize.fuse_a import (
    _merge_intervals,
    dedupe_overlapping_turns,
    subtract_coverage,
)
from fusion_diarize.types import AsrStatus, Source, Turn

DEFAULT_ABS_CAP = 12
DEFAULT_RATIO = 2.0
DEFAULT_TEXT_COLLAR = 0.5


def detect_speaker_explosion(
    moss_local_ids: set[str],
    diarizen_ids: set[str],
    *,
    abs_cap: int = DEFAULT_ABS_CAP,
    ratio: float = DEFAULT_RATIO,
) -> bool:
    """True when MOSS invents far more local speakers than DiariZen."""
    n_moss = len(moss_local_ids)
    n_dz = max(len(diarizen_ids), 1)
    return n_moss > max(abs_cap, int(ratio * n_dz))


def _attach_moss_text(
    backbone: list[Turn],
    moss_remapped: list[Turn],
    *,
    text_collar: float,
) -> list[Turn]:
    """Attach MOSS provisional text to nearest same-speaker backbone turn."""
    out = [
        Turn(
            t.start,
            t.end,
            t.speaker_id,
            t.text,
            t.asr_status,
            t.source,
            t.confidence,
        )
        for t in backbone
    ]
    for mt in moss_remapped:
        if not mt.text:
            continue
        best = None
        best_d = 1e9
        for t in out:
            if t.speaker_id != mt.speaker_id:
                continue
            d = abs(t.start - mt.start) + abs(t.end - mt.end)
            if d < best_d:
                best_d, best = d, t
        if best is not None and best_d <= 2 * text_collar:
            best.text = mt.text
            best.asr_status = AsrStatus.PROVISIONAL
            best.source = Source.FUSED
    return out


def _drop_unmapped_locals(turns: list[Turn], diarizen_ids: set[str]) -> list[Turn]:
    if not diarizen_ids:
        return turns
    return [
        t
        for t in turns
        if not (t.speaker_id.startswith("c") and ":" in t.speaker_id)
    ]


def _moss_primary_turns(moss_remapped: list[Turn]) -> list[Turn]:
    """Keep remapped MOSS; same-ID near-duplicates from chunk overlap are deduped.

    Different-speaker overlaps (true multi-talk) are intentionally kept.
    """
    deduped = dedupe_overlapping_turns(moss_remapped)
    out: list[Turn] = []
    for t in deduped:
        out.append(
            Turn(
                t.start,
                t.end,
                t.speaker_id,
                t.text,
                AsrStatus.PROVISIONAL if t.text else AsrStatus.EMPTY,
                Source.FUSED if t.text else Source.MOSS,
                t.confidence,
            )
        )
    out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
    return out


def _incomplete_spans(moss_meta: list[dict]) -> list[tuple[float, float]]:
    spans = [
        (float(m["start"]), float(m["end"]))
        for m in moss_meta
        if m.get("incomplete") or not m.get("ok", True)
    ]
    return _merge_intervals(spans)


def _intersect_with_spans(
    turn: Turn, spans: list[tuple[float, float]]
) -> list[Turn]:
    """Clip ``turn`` to the union of ``spans``."""
    out: list[Turn] = []
    for s, e in spans:
        a = max(turn.start, s)
        b = min(turn.end, e)
        if b - a > 1e-3:
            out.append(
                Turn(
                    a,
                    b,
                    turn.speaker_id,
                    "",
                    AsrStatus.EMPTY,
                    Source.DIARIZEN,
                    1.0,
                )
            )
    return out


def _moss_speech_mask(moss_turns: list[Turn]) -> list[tuple[float, float]]:
    """Union of all MOSS intervals (any speaker) — regions DiariZen must not enter."""
    return _merge_intervals([(t.start, t.end) for t in moss_turns])


def _gapfill(
    diarizen: list[Turn],
    moss_remapped: list[Turn],
    incomplete_spans: list[tuple[float, float]],
) -> list[Turn]:
    """Keep all MOSS; add DiariZen only in incomplete spans with no MOSS speech.

    Exclusivity: DiariZen is subtracted against the **union** of MOSS intervals
    (all speakers), so DiariZen and MOSS never co-label the same time.
    """
    moss_kept = _drop_unmapped_locals(
        _moss_primary_turns(moss_remapped),
        {t.speaker_id for t in diarizen},
    )
    moss_mask = _moss_speech_mask(moss_kept)

    out = list(moss_kept)
    if not incomplete_spans:
        out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
        return out

    for t in diarizen:
        for piece in _intersect_with_spans(t, incomplete_spans):
            # Remove any time already claimed by MOSS (any speaker).
            remnants = subtract_coverage(piece, moss_mask)
            out.extend(remnants)
    out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
    return out


def fuse_mode_c(
    diarizen: list[Turn],
    moss_raw: list[Turn],
    moss_remapped: list[Turn],
    moss_meta: list[dict],
    *,
    text_collar: float = DEFAULT_TEXT_COLLAR,
    abs_cap: int = DEFAULT_ABS_CAP,
    ratio: float = DEFAULT_RATIO,
) -> tuple[list[Turn], dict[str, Any]]:
    """MOSS-primary fuse with incomplete gap-fill and speaker-explosion backbone.

    Returns ``(turns, meta)`` where meta includes ``fusion_path``, counts, ``explosion``.
    """
    moss_local_ids = {t.speaker_id for t in moss_raw}
    diarizen_ids = {t.speaker_id for t in diarizen}
    n_moss = len(moss_local_ids)
    n_dz = len(diarizen_ids)
    explosion = detect_speaker_explosion(
        moss_local_ids, diarizen_ids, abs_cap=abs_cap, ratio=ratio
    )

    base_meta: dict[str, Any] = {
        "n_moss_local": n_moss,
        "n_diarizen": n_dz,
        "explosion": explosion,
    }

    if explosion:
        # Single-system path: DiariZen only (MOSS used for text attach).
        backbone = [
            Turn(
                t.start,
                t.end,
                t.speaker_id,
                "",
                AsrStatus.EMPTY,
                Source.DIARIZEN,
                1.0,
            )
            for t in diarizen
        ]
        out = _attach_moss_text(backbone, moss_remapped, text_collar=text_collar)
        out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
        base_meta["fusion_path"] = "diarizen_backbone_explosion"
        return out, base_meta

    incomplete = _incomplete_spans(moss_meta)
    if incomplete:
        out = _gapfill(diarizen, moss_remapped, incomplete)
        base_meta["fusion_path"] = "moss_primary_gapfill"
        base_meta["incomplete_spans"] = [
            {"start": a, "end": b} for a, b in incomplete
        ]
        return out, base_meta

    out = _drop_unmapped_locals(_moss_primary_turns(moss_remapped), diarizen_ids)
    base_meta["fusion_path"] = "moss_primary"
    return out, base_meta
