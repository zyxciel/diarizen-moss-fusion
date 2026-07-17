from __future__ import annotations

from fusion_diarize.types import Turn, AsrStatus, Source


def fuse_mode_a(
    diarizen: list[Turn],
    moss_remapped: list[Turn],
    confidences: dict[str, float],
    tau: float = 0.6,
) -> list[Turn]:
    """High-conf MOSS speakers: use MOSS timing+text; else DiariZen; keep unmatched."""
    high = {s for s, c in confidences.items() if c >= tau}
    out: list[Turn] = []
    for t in moss_remapped:
        if t.speaker_id in high:
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
    for t in diarizen:
        if t.speaker_id in high:
            continue  # replaced by MOSS for that speaker (simple v1)
        out.append(
            Turn(t.start, t.end, t.speaker_id, "", AsrStatus.EMPTY, Source.DIARIZEN, 1.0)
        )
    # MOSS-only low-conf already skipped; keep unmatched MOSS speakers
    diarizen_spk = {x.speaker_id for x in diarizen}
    for t in moss_remapped:
        if t.speaker_id not in high and t.speaker_id not in diarizen_spk:
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
    out.sort(key=lambda x: (x.start, x.end))
    return out
