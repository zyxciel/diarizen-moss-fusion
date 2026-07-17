# fusion_diarize/export.py
from __future__ import annotations
import json
from pathlib import Path
from fusion_diarize.types import Turn, DiarResult


def write_rttm(turns: list[Turn], path: Path, uri: str = "uri") -> None:
    lines = []
    for t in turns:
        dur = max(0.0, t.end - t.start)
        lines.append(
            f"SPEAKER {uri} 1 {t.start:.3f} {dur:.3f} <NA> <NA> {t.speaker_id} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_rttm_as_turns(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        parts = line.split()
        # SPEAKER uri 1 start dur <NA> <NA> spk ...
        start = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        turns.append(Turn(start, start + dur, spk))
    return turns


def write_json(result: DiarResult, path: Path) -> None:
    payload = {
        "meta": result.meta,
        "centroids": result.centroids,
        "turns": [t.to_dict() for t in result.turns],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> DiarResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns = [Turn.from_dict(t) for t in payload.get("turns", [])]
    return DiarResult(
        turns=turns,
        meta=payload.get("meta", {}),
        centroids=payload.get("centroids", {}),
    )
