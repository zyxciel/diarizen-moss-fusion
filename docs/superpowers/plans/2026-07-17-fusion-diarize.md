# Fusion Diarize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-only `fusion_diarize` package that runs DiariZen (`BUT-FIT/diarizen-wavlm-large-s80-md`) full-file and chunked MOSS, fuses with Mode A and Mode B, and emits RTTM + JSON (with provisional ASR fields) for remote DER experiments.

**Architecture:** Thin orchestration package at workspace root wraps vendored DiariZen and MOSS. DiariZen provides global speaker IDs + WeSpeaker centroids; hybrid chunker feeds MOSS; mapper aligns MOSS `Sxx` to DiariZen speakers; Mode A and Mode B are selectable fuse strategies sharing cached intermediates.

**Tech Stack:** Python 3.10+, PyTorch, DiariZen/pyannote, MOSS-Transcribe-Diarize, numpy, pytest. Spec: `docs/superpowers/specs/2026-07-17-fusion-diarize-design.md`.

**Defaults (locked):** chunk target 30–60 min, hard cap 90 min, overlap 30–60 s, prefer ≤8 local speakers, CLI-only, DiariZen hub `BUT-FIT/diarizen-wavlm-large-s80-md`.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `fusion_diarize/__init__.py` | Package version |
| `fusion_diarize/types.py` | `Turn`, `DiarResult`, `ChunkWindow`, `AsrStatus`, `Source` |
| `fusion_diarize/audio_prep.py` | Load/resample to 16 kHz mono; duration |
| `fusion_diarize/export.py` | RTTM + JSON read/write |
| `fusion_diarize/chunk_planner.py` | Hybrid windows from DiariZen turns |
| `fusion_diarize/mapper.py` | Overlap + cosine assignment MOSS→DiariZen |
| `fusion_diarize/fuse_a.py` | Mode A fuse |
| `fusion_diarize/fuse_b.py` | Mode B frame vote fuse |
| `fusion_diarize/diarizen_runner.py` | DiariZen with centroids + shared WeSpeaker embed for crops |
| `fusion_diarize/moss_runner.py` | Per-chunk MOSS → turns with provisional text |
| `fusion_diarize/pipeline.py` | Orchestrate cache + modes |
| `fusion_diarize/eval_der.py` | Simple DER (collar) vs ref RTTM |
| `fusion_diarize/cli.py` | `run` / `eval` entrypoints |
| `tests/test_*.py` | Synthetic unit tests (no GPU) |
| `docs/remote_smoke.md` | Copy-to-server smoke commands |
| `pyproject.toml` | Optional install of `fusion_diarize` |

Do **not** fork MOSS. Prefer subclass/wrap DiariZen in `diarizen_runner.py` rather than editing upstream permanently; if a one-line upstream fix is required, keep it minimal and documented.

---

### Task 1: Types + export (RTTM/JSON)

**Files:**
- Create: `fusion_diarize/__init__.py`
- Create: `fusion_diarize/types.py`
- Create: `fusion_diarize/export.py`
- Create: `tests/test_export.py`
- Create: `pyproject.toml` (minimal)

- [ ] **Step 1: Write failing tests for export round-trip**

```python
# tests/test_export.py
from fusion_diarize.types import Turn, DiarResult, AsrStatus, Source
from fusion_diarize.export import write_rttm, write_json, read_json, read_rttm_as_turns

def test_json_roundtrip(tmp_path):
    turns = [
        Turn(0.0, 1.0, "speaker_0", text="hi", asr_status=AsrStatus.PROVISIONAL,
             source=Source.FUSED, confidence=0.9),
        Turn(1.5, 2.0, "speaker_1", text="", asr_status=AsrStatus.EMPTY,
             source=Source.DIARIZEN, confidence=0.5),
    ]
    result = DiarResult(turns=turns, meta={"mode": "a"})
    path = tmp_path / "out.json"
    write_json(result, path)
    loaded = read_json(path)
    assert len(loaded.turns) == 2
    assert loaded.turns[0].speaker_id == "speaker_0"
    assert loaded.turns[0].text == "hi"
    assert loaded.meta["mode"] == "a"

def test_rttm_roundtrip(tmp_path):
    turns = [Turn(1.0, 3.5, "speaker_0")]
    path = tmp_path / "out.rttm"
    write_rttm(turns, path, uri="utt1")
    loaded = read_rttm_as_turns(path)
    assert len(loaded) == 1
    assert abs(loaded[0].start - 1.0) < 1e-6
    assert abs(loaded[0].end - 3.5) < 1e-6
    assert loaded[0].speaker_id == "speaker_0"
```

- [ ] **Step 2: Run tests — expect fail (import error)**

```bash
cd /Users/zhangyuxiang/Work/huawei26Q2/diarizen+moss
PYTHONPATH=. pytest tests/test_export.py -v
```

Expected: FAIL — `ModuleNotFoundError: fusion_diarize`

- [ ] **Step 3: Implement types + export**

```python
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
```

```python
# fusion_diarize/export.py
from __future__ import annotations
import json
from pathlib import Path
from fusion_diarize.types import Turn, DiarResult, AsrStatus, Source

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
```

```python
# fusion_diarize/__init__.py
__version__ = "0.1.0"
```

```toml
# pyproject.toml
[project]
name = "fusion-diarize"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
fusion-diarize = "fusion_diarize.cli:main"

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. pytest tests/test_export.py -v
```

Expected: PASS

- [ ] **Step 5: Commit** (only if user asked / repo is git-initialized)

```bash
git add fusion_diarize tests/test_export.py pyproject.toml
git commit -m "feat: add fusion_diarize turn types and RTTM/JSON export"
```

---

### Task 2: Audio prep

**Files:**
- Create: `fusion_diarize/audio_prep.py`
- Create: `tests/test_audio_prep.py`

- [ ] **Step 1: Write failing test (synthetic wav via torchaudio or soundfile; if torchaudio unavailable in unit env, test duration helper with mocked load)**

```python
# tests/test_audio_prep.py
import numpy as np
from pathlib import Path
from fusion_diarize.audio_prep import probe_duration, write_mono16k_wav, load_mono16k

def test_write_and_probe(tmp_path):
    path = tmp_path / "a.wav"
    # 1.5 seconds of silence at 16k
    write_mono16k_wav(path, np.zeros(24000, dtype=np.float32))
    assert abs(probe_duration(path) - 1.5) < 1e-3
    wav, sr = load_mono16k(path)
    assert sr == 16000
    assert wav.ndim == 1
    assert len(wav) == 24000
```

- [ ] **Step 2: Run — expect fail**

```bash
PYTHONPATH=. pytest tests/test_audio_prep.py -v
```

- [ ] **Step 3: Implement with torchaudio (preferred) or scipy+wave fallback**

```python
# fusion_diarize/audio_prep.py
from __future__ import annotations
from pathlib import Path
import numpy as np

def write_mono16k_wav(path: Path, waveform: np.ndarray) -> None:
    import torch
    import torchaudio
    assert waveform.ndim == 1
    tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
    torchaudio.save(str(path), tensor, 16000)

def load_mono16k(path: Path) -> tuple[np.ndarray, int]:
    import torch
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0).numpy(), sr

def probe_duration(path: Path) -> float:
    wav, sr = load_mono16k(path)
    return float(len(wav) / sr)

def slice_wav(path: Path, start: float, end: float, out_path: Path) -> Path:
    wav, sr = load_mono16k(path)
    i0 = max(0, int(round(start * sr)))
    i1 = min(len(wav), int(round(end * sr)))
    write_mono16k_wav(out_path, wav[i0:i1])
    return out_path
```

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit** if requested

---

### Task 3: Chunk planner (hybrid)

**Files:**
- Create: `fusion_diarize/chunk_planner.py`
- Create: `tests/test_chunk_planner.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_chunk_planner.py
from fusion_diarize.types import Turn
from fusion_diarize.chunk_planner import plan_chunks

def test_short_audio_single_chunk():
    turns = [Turn(0.0, 10.0, "speaker_0")]
    chunks = plan_chunks(turns, duration=120.0, target_min=1800, target_max=3600, hard_cap=5400)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 120.0

def test_long_audio_splits_near_gap():
    # 2 hours; silence gap around 3500s
    turns = [
        Turn(0.0, 3400.0, "speaker_0"),
        Turn(3600.0, 7000.0, "speaker_1"),
    ]
    chunks = plan_chunks(
        turns, duration=7200.0,
        target_min=1800, target_max=3600, hard_cap=5400,
        overlap=30.0, max_local_speakers=8,
    )
    assert len(chunks) >= 2
    assert all(c.end - c.start <= 5400 + 1e-6 for c in chunks)
    # cut should be near the gap, not mid-speech if possible
    assert any(3300 <= c.end <= 3700 for c in chunks[:-1])

def test_high_speaker_density_flag():
    turns = [Turn(0.0, 100.0, f"speaker_{i}") for i in range(10)]
    chunks = plan_chunks(turns, duration=100.0, target_min=1800, target_max=3600, hard_cap=5400)
    assert chunks[0].high_speaker_density is True
    assert chunks[0].n_local_speakers >= 9
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement planner**

Logic:
1. If `duration <= target_max`, return one chunk `[0, duration]`.
2. Else find candidate cut times in `[target_min, min(target_max, hard_cap)]` from start of current window: prefer midpoints of gaps between DiariZen turns longer than 0.5 s; else cut at `target_max`.
3. Advance with `overlap` seconds backward for next window start.
4. For each window, count distinct DiariZen speakers overlapping the window; set `high_speaker_density = (n > max_local_speakers)`.

```python
# fusion_diarize/chunk_planner.py
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
        if start >= end - 1.0:  # safety
            start = end
    return chunks
```

- [ ] **Step 4: pytest pass** (adjust cut assertion if gap midpoint lands slightly differently — keep invariant `end-start <= hard_cap`)

- [ ] **Step 5: Commit** if requested

---

### Task 4: Mapper (overlap + embedding)

**Files:**
- Create: `fusion_diarize/mapper.py`
- Create: `tests/test_mapper.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_mapper.py
import numpy as np
from fusion_diarize.types import Turn
from fusion_diarize.mapper import map_moss_speakers, assignment_confidence

def test_map_by_overlap_and_embedding():
    diarizen = [
        Turn(0.0, 5.0, "speaker_0"),
        Turn(5.0, 10.0, "speaker_1"),
    ]
    moss = [
        Turn(0.2, 4.8, "S01", text="a"),
        Turn(5.2, 9.5, "S02", text="b"),
    ]
    centroids = {
        "speaker_0": np.array([1.0, 0.0]),
        "speaker_1": np.array([0.0, 1.0]),
    }
    moss_emb = {
        "S01": np.array([0.9, 0.1]),
        "S02": np.array([0.1, 0.9]),
    }
    mapping = map_moss_speakers(moss, diarizen, moss_emb, centroids)
    assert mapping["S01"] == "speaker_0"
    assert mapping["S02"] == "speaker_1"

def test_low_confidence_when_ambiguous():
    # equal overlap and similar embeddings -> low confidence
    conf = assignment_confidence(overlap=0.1, cosine=0.2)
    assert conf < 0.5
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

```python
# fusion_diarize/mapper.py
from __future__ import annotations
import numpy as np
from fusion_diarize.types import Turn

def _overlap(a: Turn, b: Turn) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def assignment_confidence(overlap: float, cosine: float, overlap_scale: float = 5.0) -> float:
    o = min(1.0, overlap / overlap_scale)
    c = max(0.0, min(1.0, (cosine + 1.0) / 2.0))  # map [-1,1] -> [0,1]
    return 0.5 * o + 0.5 * c

def map_moss_speakers(
    moss_turns: list[Turn],
    diarizen_turns: list[Turn],
    moss_emb: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> dict[str, str]:
    """Greedy: for each local MOSS label, score global DiariZen speakers."""
    local_labels = sorted({t.speaker_id for t in moss_turns})
    global_labels = list(centroids.keys()) or sorted({t.speaker_id for t in diarizen_turns})
    mapping: dict[str, str] = {}
    used_global: set[str] = set()
    scores: list[tuple[float, str, str]] = []
    for loc in local_labels:
        loc_turns = [t for t in moss_turns if t.speaker_id == loc]
        for glob in global_labels:
            glob_turns = [t for t in diarizen_turns if t.speaker_id == glob]
            ov = sum(_overlap(a, b) for a in loc_turns for b in glob_turns)
            cos = 0.0
            if loc in moss_emb and glob in centroids:
                cos = cosine(moss_emb[loc], centroids[glob])
            score = assignment_confidence(ov, cos)
            scores.append((score, loc, glob))
    scores.sort(reverse=True)
    for score, loc, glob in scores:
        if loc in mapping or glob in used_global:
            continue
        mapping[loc] = glob
        used_global.add(glob)
    # leftover locals: best glob even if reused
    for loc in local_labels:
        if loc in mapping:
            continue
        best = None
        best_s = -1.0
        for score, l, glob in scores:
            if l == loc and score > best_s:
                best_s, best = score, glob
        if best is not None:
            mapping[loc] = best
    return mapping

def remap_turns(moss_turns: list[Turn], mapping: dict[str, str]) -> list[Turn]:
    out = []
    for t in moss_turns:
        spk = mapping.get(t.speaker_id, t.speaker_id)
        out.append(Turn(t.start, t.end, spk, t.text, t.asr_status, t.source, t.confidence))
    return out
```

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit** if requested

---

### Task 5: Fuse Mode A and Mode B

**Files:**
- Create: `fusion_diarize/fuse_a.py`
- Create: `fusion_diarize/fuse_b.py`
- Create: `tests/test_fuse.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_fuse.py
from fusion_diarize.types import Turn, AsrStatus, Source
from fusion_diarize.fuse_a import fuse_mode_a
from fusion_diarize.fuse_b import fuse_mode_b

def test_mode_a_prefers_moss_boundaries_when_confident():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL, source=Source.MOSS)]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.9}, tau=0.6)
    assert len(fused) == 1
    assert abs(fused[0].start - 0.1) < 1e-6
    assert fused[0].text == "hello"
    assert fused[0].source == Source.FUSED

def test_mode_a_fallback_diarizen_on_low_conf():
    diarizen = [Turn(0.0, 5.0, "speaker_0")]
    moss = [Turn(0.1, 4.9, "speaker_0", text="hello", asr_status=AsrStatus.PROVISIONAL)]
    fused = fuse_mode_a(diarizen, moss, confidences={"speaker_0": 0.2}, tau=0.6)
    assert fused[0].start == 0.0
    assert fused[0].source == Source.DIARIZEN

def test_mode_b_voting_keeps_speech():
    diarizen = [Turn(0.0, 2.0, "speaker_0")]
    moss = [Turn(1.0, 3.0, "speaker_0", text="x", asr_status=AsrStatus.PROVISIONAL)]
    fused = fuse_mode_b(diarizen, moss, frame_hop=0.2)
    assert any(t.speaker_id == "speaker_0" for t in fused)
    assert max(t.end for t in fused) >= 2.5
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement fuse_a and fuse_b**

```python
# fusion_diarize/fuse_a.py
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
    moss_spk = {t.speaker_id for t in moss_remapped}
    for t in moss_remapped:
        if t.speaker_id in high:
            out.append(Turn(
                t.start, t.end, t.speaker_id, t.text,
                AsrStatus.PROVISIONAL if t.text else AsrStatus.EMPTY,
                Source.FUSED, confidences.get(t.speaker_id, 1.0),
            ))
    for t in diarizen:
        if t.speaker_id in high:
            continue  # replaced by MOSS for that speaker (simple v1)
        out.append(Turn(t.start, t.end, t.speaker_id, "", AsrStatus.EMPTY, Source.DIARIZEN, 1.0))
    # MOSS-only low-conf already skipped; optionally keep low-conf moss flagged
    for t in moss_remapped:
        if t.speaker_id not in high and t.speaker_id not in {x.speaker_id for x in diarizen}:
            out.append(Turn(
                t.start, t.end, t.speaker_id, t.text,
                AsrStatus.PROVISIONAL if t.text else AsrStatus.EMPTY,
                Source.MOSS, confidences.get(t.speaker_id, 0.3),
            ))
    out.sort(key=lambda x: (x.start, x.end))
    return out
```

```python
# fusion_diarize/fuse_b.py
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
        turns = []
        i = 0
        while i < len(grid):
            if grid[i] == 0:
                i += 1
                continue
            j = i
            while j < len(grid) and grid[j] > 0:
                j += 1
            src = Source.FUSED if grid[i:j].max() == 2 else Source.DIARIZEN
            turns.append(Turn(i * frame_hop, j * frame_hop, spk, "", AsrStatus.EMPTY, src, 1.0))
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
```

Note: Mode A v1 replaces entire DiariZen tracks for high-conf speakers with MOSS tracks (simple). If experiments show gaps, refine to segment-level merge in a follow-up — do not expand in v1 unless DER requires it.

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit** if requested

---

### Task 6: DiariZen runner (centroids + crop embed)

**Files:**
- Create: `fusion_diarize/diarizen_runner.py`
- Create: `tests/test_diarizen_runner_unit.py` (mock-friendly helpers only; GPU smoke documented separately)

- [ ] **Step 1: Implement runner that subclasses/wraps DiariZen and keeps centroids**

Key change vs upstream `DiariZenPipeline.__call__` (`DiariZen-main/diarizen/pipelines/inference.py` ~153):

```python
hard_clusters, _, centroids = self.clustering(
    embeddings=embeddings,
    segmentations=binarized_segmentations,
    min_clusters=self.min_speakers,
    max_clusters=self.max_speakers,
)
```

Return `(annotation, centroids_dict)` where keys match annotation labels (`speaker_0`, …). Align centroid rows to labels the same way pyannote `apply(..., return_embeddings=True)` does (`speaker_diarization.py` ~612–636): reorder by label list order.

Also expose `embed_crop(waveform_16k, start, end) -> np.ndarray` using `pipeline._embedding` / WeSpeaker already loaded on the pipeline — same model, no second hub download.

```python
# fusion_diarize/diarizen_runner.py (sketch)
from __future__ import annotations
from pathlib import Path
import numpy as np
from fusion_diarize.types import Turn
from fusion_diarize.audio_prep import load_mono16k

DIARIZEN_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md"

class DiariZenRunner:
    def __init__(self, repo_id: str = DIARIZEN_REPO, cache_dir: str | None = None, device: str | None = None):
        # Import only when constructed (heavy)
        from diarizen.pipelines.inference import DiariZenPipeline
        self.pipeline = DiariZenPipeline.from_pretrained(repo_id, cache_dir=cache_dir)
        # optional: move to device if API allows

    def run(self, audio_path: Path, sess_name: str = "utt") -> tuple[list[Turn], dict[str, np.ndarray]]:
        annotation, centroids = self._call_with_centroids(audio_path, sess_name)
        turns = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append(Turn(float(segment.start), float(segment.end), str(label)))
        # centroids: np.ndarray (num_spk, dim) aligned to annotation.labels()
        labels = list(annotation.labels())
        cent = {}
        if centroids is not None:
            for i, lab in enumerate(labels):
                if i < len(centroids):
                    cent[str(lab)] = np.asarray(centroids[i], dtype=np.float32)
        return turns, cent

    def embed_moss_labels(
        self, audio_path: Path, moss_turns: list[Turn]
    ) -> dict[str, np.ndarray]:
        """Mean-pool WeSpeaker embeddings over all crops for each local speaker label."""
        wav, sr = load_mono16k(audio_path)
        assert sr == 16000
        by_label: dict[str, list[np.ndarray]] = {}
        for t in moss_turns:
            emb = self._embed_region(wav, t.start, t.end)
            by_label.setdefault(t.speaker_id, []).append(emb)
        return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in by_label.items()}

    def _call_with_centroids(self, audio_path, sess_name):
        # Copy DiariZenPipeline.__call__ body but keep centroids; return annotation, centroids
        ...

    def _embed_region(self, wav: np.ndarray, start: float, end: float) -> np.ndarray:
        # Use pipeline embedding model on crop; if API is batch-oriented,
        # pass a single chunk with full-activity mask. Keep implementation minimal.
        ...
```

- [ ] **Step 2: Unit-test only pure helpers** (label alignment) without loading HF models:

```python
def test_align_centroids_to_labels():
    from fusion_diarize.diarizen_runner import align_centroids
    labels = ["speaker_1", "speaker_0"]
    cents = np.array([[0.0, 1.0], [1.0, 0.0]])
    # after align to sorted annotation label order used by runner
    d = align_centroids(labels, cents)
    assert set(d) == {"speaker_0", "speaker_1"}
```

- [ ] **Step 3: Document GPU smoke in `docs/remote_smoke.md`** (filled in Task 9)

- [ ] **Step 4: Commit** if requested

---

### Task 7: MOSS runner (per chunk)

**Files:**
- Create: `fusion_diarize/moss_runner.py`

- [x] **Step 1: Implement wrapper around MOSS `generate_transcription` + `parse_transcript`**

```python
# fusion_diarize/moss_runner.py
from __future__ import annotations
from pathlib import Path
from fusion_diarize.types import Turn, AsrStatus, Source
from fusion_diarize.types import ChunkWindow

class MossRunner:
    def __init__(self, model_path: str, device: str = "cuda"):
        # load HF model/processor once; follow MOSS README Quickstart
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # from transformers import ... Moss classes as in MOSS README
        ...

    def transcribe_chunk(self, chunk_wav: Path, time_offset: float) -> list[Turn]:
        self._ensure_loaded()
        from moss_transcribe_diarize.inference_utils import generate_transcription
        from moss_transcribe_diarize.transcript_parser import parse_transcript
        text = generate_transcription(...)  # per MOSS API
        segs = parse_transcript(text)
        turns = []
        for s in segs:
            turns.append(Turn(
                start=float(s.start) + time_offset,
                end=float(s.end) + time_offset,
                speaker_id=str(s.speaker),  # e.g. S01
                text=str(s.text),
                asr_status=AsrStatus.PROVISIONAL if s.text else AsrStatus.EMPTY,
                source=Source.MOSS,
                confidence=1.0,
            ))
        return turns

    def run_chunks(self, full_audio: Path, chunks: list[ChunkWindow], work_dir: Path) -> tuple[list[Turn], list[dict]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        all_turns: list[Turn] = []
        meta = []
        from fusion_diarize.audio_prep import slice_wav
        for i, c in enumerate(chunks):
            chunk_path = work_dir / f"chunk_{i:03d}.wav"
            try:
                slice_wav(full_audio, c.start, c.end, chunk_path)
                turns = self.transcribe_chunk(chunk_path, time_offset=c.start)
                all_turns.extend(turns)
                meta.append({"chunk": i, "start": c.start, "end": c.end, "ok": True,
                             "high_speaker_density": c.high_speaker_density})
            except Exception as e:
                meta.append({"chunk": i, "start": c.start, "end": c.end, "ok": False,
                             "error": str(e), "high_speaker_density": c.high_speaker_density})
        return all_turns, meta
```

Wire exact `generate_transcription` kwargs from MOSS README at implementation time (`model`, `processor`, audio path, `max_new_tokens` raised for long chunks).

- [x] **Step 2: No GPU unit test required; add a fake runner interface for pipeline tests**

```python
class FakeMossRunner:
    def run_chunks(self, full_audio, chunks, work_dir):
        return [], [{"chunk": 0, "ok": True}]
```

- [ ] **Step 3: Commit** if requested

---

### Task 8: Pipeline + CLI + caching

**Files:**
- Create: `fusion_diarize/pipeline.py`
- Create: `fusion_diarize/cli.py`
- Create: `tests/test_pipeline_offline.py`

- [x] **Step 1: `work_dir` layout**

```text
work_dir/
  prepared.wav
  diarizen.json          # turns + centroids
  chunks.json            # ChunkWindow list
  moss/chunk_XXX.wav
  moss_turns.json
  mode_a.rttm / mode_a.json
  mode_b.rttm / mode_b.json
```

- [x] **Step 2: Pipeline**

```python
# fusion_diarize/pipeline.py
def run_pipeline(audio, work_dir, mode: str, diarizen_runner, moss_runner, tau=0.6):
    # 1. prep -> prepared.wav (cache if exists)
    # 2. diarizen.run -> cache diarizen.json
    # 3. plan_chunks -> chunks.json
    # 4. moss.run_chunks -> moss_turns.json (skip if cached)
    # 5. embed moss labels via diarizen_runner.embed_moss_labels
    # 6. map_moss_speakers + per-label confidence
    # 7. if mode in (a, both): fuse_mode_a -> export
    # 8. if mode in (b, both): fuse_mode_b -> export
```

Offline test: inject FakeDiariZen + FakeMoss with fixed turns; assert exports exist for `--mode both`.

- [x] **Step 3: CLI**

```python
# fusion_diarize/cli.py
def main():
    import argparse
    p = argparse.ArgumentParser(prog="fusion-diarize")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--audio", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--mode", choices=["a", "b", "both"], default="both")
    run.add_argument("--moss-model", required=True)
    run.add_argument("--diarizen-repo", default="BUT-FIT/diarizen-wavlm-large-s80-md")
    run.add_argument("--tau", type=float, default=0.6)

    ev = sub.add_parser("eval")
    ev.add_argument("--hyp", required=True)
    ev.add_argument("--ref", required=True)
    ev.add_argument("--collar", type=float, default=0.25)

    args = p.parse_args()
    if args.cmd == "run":
        ...
    elif args.cmd == "eval":
        from fusion_diarize.eval_der import compute_der
        print(compute_der(args.hyp, args.ref, collar=args.collar))
```

- [x] **Step 4: pytest offline pipeline**

- [ ] **Step 5: Commit** if requested

---

### Task 9: DER eval + remote smoke docs

**Files:**
- Create: `fusion_diarize/eval_der.py`
- Create: `tests/test_eval_der.py`
- Create: `docs/remote_smoke.md`

- [ ] **Step 1: Simple DER with collar** (pure numpy; or call `pyannote.metrics` if already installed with DiariZen)

```python
# Prefer pyannote.metrics.diarization.DiarizationErrorRate when available.
# Fallback: document that remote env should use pyannote from DiariZen deps.
```

Test: identical hyp/ref → DER 0; clear miss → DER > 0.

- [ ] **Step 2: `docs/remote_smoke.md`**

Contents:
1. Copy workspace to server
2. Install DiariZen + MOSS deps (point to each README)
3. `pip install -e .` for fusion_diarize
4. Example:

```bash
fusion-diarize run \
  --audio /data/utt.wav \
  --work-dir /data/work/utt \
  --mode both \
  --moss-model /models/MOSS-Transcribe-Diarize

fusion-diarize eval --hyp /data/work/utt/mode_a.rttm --ref /data/ref/utt.rttm
fusion-diarize eval --hyp /data/work/utt/mode_b.rttm --ref /data/ref/utt.rttm
```

5. Baselines: export DiariZen-only and MOSS-remapped-only from cached intermediates for ablation.

- [ ] **Step 3: pytest eval**

- [ ] **Step 4: Commit** if requested

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| DiariZen s80-md full-file | Task 6 |
| Expose WeSpeaker centroids; embed MOSS crops same model | Task 6 |
| Hybrid chunking + density flag | Task 3 |
| MOSS per chunk + provisional text | Task 7 |
| Mapper overlap+cosine | Task 4 |
| Mode A + Mode B | Task 5 |
| RTTM + JSON + asr_status | Task 1 |
| CLI-only run/eval + cache | Task 8–9 |
| DER experiments / remote | Task 9 |
| MOSS failure → DiariZen fallback | Task 7–8 meta + fuse uses diarizen spans |
| No UI | honored |

## Self-review notes

- No TBD left in task steps; MOSS `generate_transcription` exact kwargs filled from README at Task 7 implementation (API is stable in-tree).
- Mode A v1 is speaker-track replacement for high-conf labels (simple); documented follow-up if DER needs segment-level merge.
- Types (`Turn`, `AsrStatus`, `Source`) consistent across tasks.
- Commits only when user requests (workspace root may not be a git repo).

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-17-fusion-diarize.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
