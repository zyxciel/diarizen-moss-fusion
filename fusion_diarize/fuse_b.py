from __future__ import annotations

import numpy as np

from fusion_diarize.types import Turn, AsrStatus, Source


def fuse_mode_b(
    diarizen: list[Turn],
    moss_remapped: list[Turn],
    frame_hop: float = 0.02,
    text_collar: float = 0.5,
) -> list[Turn]:
    if not diarizen and not moss_remapped:
        return []
    t_max = max([t.end for t in diarizen + moss_remapped], default=0.0)
    n = int(np.ceil(t_max / frame_hop)) + 1
    # speaker -> frame activity preference: 2=moss, 1=diarizen
    speakers = sorted({t.speaker_id for t in diarizen + moss_remapped})
    grids = {s: np.zeros(n, dtype=np.int8) for s in speakers}
    for t in diarizen:
        i0, i1 = int(t.start / frame_hop), int(t.end / frame_hop)
        grids[t.speaker_id][i0:i1] = np.maximum(grids[t.speaker_id][i0:i1], 1)
    for t in moss_remapped:
        i0, i1 = int(t.start / frame_hop), int(t.end / frame_hop)
        grids[t.speaker_id][i0:i1] = np.maximum(grids[t.speaker_id][i0:i1], 2)

    def frames_to_turns(spk: str, grid: np.ndarray) -> list[Turn]:
        turns: list[Turn] = []
        i = 0
        while i < len(grid):
            if grid[i] == 0:
                i += 1
                continue
            j = i
            while j < len(grid) and grid[j] > 0:
                j += 1
            src = Source.FUSED if grid[i:j].max() == 2 else Source.DIARIZEN
            turns.append(
                Turn(i * frame_hop, j * frame_hop, spk, "", AsrStatus.EMPTY, src, 1.0)
            )
            i = j
        return turns

    out: list[Turn] = []
    for spk in speakers:
        out.extend(frames_to_turns(spk, grids[spk]))

    # attach moss text to nearest fused turn of same speaker
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
    out.sort(key=lambda x: (x.start, x.end))
    return out
