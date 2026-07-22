"""Mode C fusion: MOSS-primary with DiariZen gap-fill and explosion guard."""
from __future__ import annotations

from typing import Any

from fusion_diarize.fuse_a import dedupe_overlapping_turns, subtract_coverage
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


def _moss_primary_turns(moss_remapped: list[Turn]) -> list[Turn]:
    """Keep remapped MOSS only; drop bare unmapped local IDs when DiariZen IDs exist."""
    deduped = dedupe_overlapping_turns(moss_remapped)
    out: list[Turn] = []
    for t in deduped:
        # Prefer global DiariZen-style ids; keep remapped moss with text.
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


def _gapfill(
    diarizen: list[Turn],
    moss_remapped: list[Turn],
) -> list[Turn]:
    """MOSS remapped + DiariZen remnants outside MOSS coverage (same speaker)."""
    moss_kept = _moss_primary_turns(moss_remapped)
    # Drop unmapped namespaced locals from final mix when DiariZen IDs exist.
    diarizen_ids = {t.speaker_id for t in diarizen}
    if diarizen_ids:
        moss_kept = [
            t
            for t in moss_kept
            if not (t.speaker_id.startswith("c") and ":" in t.speaker_id)
        ]

    coverage: dict[str, list[tuple[float, float]]] = {}
    for t in moss_kept:
        coverage.setdefault(t.speaker_id, []).append((t.start, t.end))

    out = list(moss_kept)
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
    out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
    return out


def _has_incomplete(moss_meta: list[dict]) -> bool:
    return any(m.get("incomplete") or not m.get("ok", True) for m in moss_meta)


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
        # Text attach needs remapped IDs aligned to DiariZen space.
        out = _attach_moss_text(backbone, moss_remapped, text_collar=text_collar)
        out.sort(key=lambda x: (x.start, x.end, x.speaker_id))
        base_meta["fusion_path"] = "diarizen_backbone_explosion"
        return out, base_meta

    if _has_incomplete(moss_meta):
        out = _gapfill(diarizen, moss_remapped)
        base_meta["fusion_path"] = "moss_primary_gapfill"
        return out, base_meta

    out = _moss_primary_turns(moss_remapped)
    # Drop unmapped locals (still namespaced cXXX:) when DiariZen speakers exist.
    if diarizen_ids:
        out = [t for t in out if not (":" in t.speaker_id and t.speaker_id.startswith("c"))]
    base_meta["fusion_path"] = "moss_primary"
    return out, base_meta
