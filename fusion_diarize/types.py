# fusion_diarize/types.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class AsrStatus(str, Enum):
    PROVISIONAL = "provisional"
    FINAL = "final"
    EMPTY = "empty"


class Source(str, Enum):
    DIARIZEN = "diarizen"
    MOSS = "moss"
    FUSED = "fused"


@dataclass
class Turn:
    start: float
    end: float
    speaker_id: str
    text: str = ""
    asr_status: AsrStatus = AsrStatus.EMPTY
    source: Source = Source.DIARIZEN
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asr_status"] = self.asr_status.value
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            speaker_id=str(d["speaker_id"]),
            text=str(d.get("text", "")),
            asr_status=AsrStatus(d.get("asr_status", "empty")),
            source=Source(d.get("source", "diarizen")),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class ChunkWindow:
    start: float
    end: float
    high_speaker_density: bool = False
    n_local_speakers: int = 0


@dataclass
class DiarResult:
    turns: list[Turn]
    meta: dict[str, Any] = field(default_factory=dict)
    # speaker_id -> embedding vector (list[float]) when available
    centroids: dict[str, list[float]] = field(default_factory=dict)
