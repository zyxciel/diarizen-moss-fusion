"""Per-chunk MOSS-Transcribe-Diarize runner → provisional ASR turns.

Vendors ``MOSS-Transcribe-Diarize-main`` onto ``sys.path`` relative to the
workspace root (parent of ``fusion_diarize/``) before lazy MOSS imports.
Does not load HF models at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fusion_diarize.types import AsrStatus, ChunkWindow, Source, Turn

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_MOSS_ROOT = _WORKSPACE_ROOT / "MOSS-Transcribe-Diarize-main"

# MOSS README recommends 65536 for long multi-speaker audio (context max_length=131072).
# Keep generation budget at the documented upper example so 20-min chunks do not truncate.
DEFAULT_MAX_NEW_TOKENS = 65536
# If last segment ends before this fraction of chunk duration → incomplete.
INCOMPLETE_COVERAGE_RATIO = 0.85


def _ensure_moss_on_path() -> None:
    """Insert vendored MOSS package root onto ``sys.path``."""
    s = str(_MOSS_ROOT)
    if _MOSS_ROOT.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


def segments_to_turns(
    segments: list[Any],
    offset: float,
    *,
    speaker_prefix: str = "",
) -> list[Turn]:
    """Convert parse_transcript-like segments to Turns with a time offset.

    ``speaker_prefix`` namespaces local IDs across chunks (e.g. ``c003:``)
    so identical ``S01`` labels from different MOSS runs do not collide.
    """
    turns: list[Turn] = []
    for s in segments:
        text = str(getattr(s, "text", "") or "")
        spk = str(s.speaker)
        if speaker_prefix:
            spk = f"{speaker_prefix}{spk}"
        turns.append(
            Turn(
                start=float(s.start) + offset,
                end=float(s.end) + offset,
                speaker_id=spk,
                text=text,
                asr_status=AsrStatus.PROVISIONAL if text else AsrStatus.EMPTY,
                source=Source.MOSS,
                confidence=1.0,
            )
        )
    return turns


def detect_incomplete(
    turns: list[Turn],
    chunk_start: float,
    chunk_end: float,
    *,
    generated_tokens: int | None = None,
    max_new_tokens: int | None = None,
    coverage_ratio: float = INCOMPLETE_COVERAGE_RATIO,
) -> tuple[bool, str]:
    """Heuristic for truncated / incomplete MOSS output on a chunk.

    Returns ``(incomplete, reason)``.
    """
    dur = chunk_end - chunk_start
    if dur <= 0:
        return False, ""
    if generated_tokens is not None and max_new_tokens is not None:
        if generated_tokens >= max_new_tokens:
            return True, "hit_max_new_tokens"
    if not turns:
        return True, "empty_output"
    last_end = max(t.end for t in turns)
    covered = last_end - chunk_start
    if covered < coverage_ratio * dur:
        return True, f"low_coverage:{covered / dur:.2f}"
    return False, ""


class MossRunner:
    """Lazy-loads MOSS HF model/processor; transcribes chunk WAVs to Turns."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._torch_device = None
        self._dtype = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_moss_on_path()
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        from moss_transcribe_diarize.inference_utils import resolve_device

        device = resolve_device(self.device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype="auto",
        )
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = model.to(dtype=dtype).to(device).eval()
        self._processor = processor
        self._torch_device = device
        self._dtype = dtype

    def transcribe_chunk(
        self,
        chunk_wav: Path,
        time_offset: float,
        chunk_index: int | None = None,
    ) -> tuple[list[Turn], dict[str, Any]]:
        """Transcribe one chunk; returns turns + generation diagnostics."""
        self._ensure_loaded()
        _ensure_moss_on_path()
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )

        messages = build_transcription_messages(chunk_wav)
        result = generate_transcription(
            self._model,
            self._processor,
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            device=self._torch_device,
            dtype=self._dtype,
        )
        segs = parse_transcript(result["text"])
        prefix = f"c{chunk_index:03d}:" if chunk_index is not None else ""
        turns = segments_to_turns(segs, time_offset, speaker_prefix=prefix)
        diag = {
            "generated_tokens": int(result.get("generated_tokens", 0)),
            "max_new_tokens": self.max_new_tokens,
        }
        return turns, diag

    def run_chunks(
        self,
        full_audio: Path,
        chunks: list[ChunkWindow],
        work_dir: Path,
    ) -> tuple[list[Turn], list[dict]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        all_turns: list[Turn] = []
        meta: list[dict] = []
        from fusion_diarize.audio_prep import slice_wav

        for i, c in enumerate(chunks):
            chunk_path = work_dir / f"chunk_{i:03d}.wav"
            try:
                slice_wav(full_audio, c.start, c.end, chunk_path)
                turns, diag = self.transcribe_chunk(
                    chunk_path, time_offset=c.start, chunk_index=i
                )
                incomplete, reason = detect_incomplete(
                    turns,
                    c.start,
                    c.end,
                    generated_tokens=diag.get("generated_tokens"),
                    max_new_tokens=diag.get("max_new_tokens"),
                )
                all_turns.extend(turns)
                meta.append(
                    {
                        "chunk": i,
                        "start": c.start,
                        "end": c.end,
                        "ok": True,
                        "high_speaker_density": c.high_speaker_density,
                        "incomplete": incomplete,
                        "incomplete_reason": reason,
                        "n_turns": len(turns),
                        "generated_tokens": diag.get("generated_tokens"),
                        "max_new_tokens": diag.get("max_new_tokens"),
                    }
                )
            except Exception as e:
                meta.append(
                    {
                        "chunk": i,
                        "start": c.start,
                        "end": c.end,
                        "ok": False,
                        "error": str(e),
                        "high_speaker_density": c.high_speaker_density,
                        "incomplete": True,
                        "incomplete_reason": "exception",
                    }
                )
        return all_turns, meta


class FakeMossRunner:
    """Offline stand-in for pipeline tests (no HF / GPU)."""

    def __init__(self, turns_by_default: list[Turn] | None = None):
        self.turns = list(turns_by_default) if turns_by_default is not None else []

    def run_chunks(
        self,
        full_audio: Path,
        chunks: list[ChunkWindow],
        work_dir: Path,
    ) -> tuple[list[Turn], list[dict]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "chunk": i,
                "start": c.start,
                "end": c.end,
                "ok": True,
                "high_speaker_density": c.high_speaker_density,
                "incomplete": False,
                "incomplete_reason": "",
            }
            for i, c in enumerate(chunks)
        ]
        return list(self.turns), meta
