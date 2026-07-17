# Fusion Diarization Pipeline Design

**Date:** 2026-07-17  
**Status:** Draft for review  
**Goal:** Best-effort offline speaker diarization annotation by combining DiariZen and MOSS-Transcribe-Diarize, with ASR fields reserved for a later stage.

## 1. Problem

We need high-quality speaker diarization labels on real meeting-style audio where:

- Audio is often **long** (beyond MOSS’s practical comfort zone of ~60 minutes; hard product limit ~90 minutes).
- Speaker count is often **>8**, where MOSS degrades.
- DiariZen (`BUT-FIT/diarizen-wavlm-large-s80-md`) has no hard duration/speaker cap and is solid (~80) but weaker than MOSS (~90) on short, few-speaker audio.

There is no existing fusion layer in this workspace; DiariZen and MOSS are independent vendored trees.

## 2. Goals and non-goals

### Goals (v1)

- Optimize **diarization** (who spoke when), not final ASR.
- Solve MOSS limits via **hybrid chunking** + cross-chunk **speaker identity** from DiariZen.
- Implement **two fusion modes (A and B)** and choose the default from **remote CLI experiments** (DER).
- Emit **RTTM + JSON**; JSON includes provisional ASR `text` from MOSS and a Stage-2 ASR hook.
- **CLI-only** delivery suitable for copying to a remote GPU server.

### Non-goals (v1)

- Review / edit UI (local or remote).
- Production serving / job queue.
- Replacing MOSS provisional text with a dedicated ASR model (schema only).
- Cross-file speaker enrollment across sessions.

## 3. Decisions locked in

| Decision | Choice |
|----------|--------|
| Primary deliverable | Diarization labels; ASR next stage with interface |
| Data profile | Worst case: long + many speakers |
| Disagreement policy | Rule-based blend: DiariZen = global identity; MOSS = local precision when trusted |
| Chunking | Hybrid: DiariZen-guided, fixed-window fallback |
| Fusion modes | Implement **both A and B**; pick via experiments |
| DiariZen model | `BUT-FIT/diarizen-wavlm-large-s80-md` |
| Embeddings | Reuse DiariZen’s WeSpeaker path (expose centroids); embed MOSS crops with the **same** WeSpeaker for mapping |
| Outputs | RTTM + JSON (JSON includes `text` / `asr_status`) |
| Interface | CLI only |

## 4. Architecture

New package at workspace root: `fusion_diarize/`. Upstream trees stay vendored; thin wrap/patch only where needed (DiariZen must not discard clustering embeddings).

```text
Audio → Prep(16k mono)
     → DiariZen full-file (s80-md) → turns + global WeSpeaker centroids
     → ChunkPlanner (hybrid)
     → MOSS per chunk → local turns + provisional text
     → Fuse Mode A and/or Mode B
     → RTTM + JSON
     → Optional DER eval / ablation
```

Shared intermediates are cached under `work_dir/` so Mode A and Mode B reuse the same DiariZen and MOSS runs.

### 4.1 Mode A — DiariZen-first identity + MOSS local refine

1. Plan chunks from DiariZen (silence/low-overlap cuts; prefer ≤60 min and ≤8 local speakers).
2. Run MOSS per chunk.
3. Map MOSS `Sxx` → DiariZen `speaker_k` using temporal overlap + cosine similarity between MOSS-crop WeSpeaker embeddings and DiariZen global centroids.
4. Fuse: high-confidence maps use MOSS boundaries + DiariZen IDs + MOSS text; low-confidence or MOSS failure falls back to DiariZen; unmatched DiariZen speakers kept; unmatched MOSS speech kept and flagged (`source=moss`).

### 4.2 Mode B — Parallel late fusion

1. Same DiariZen + chunked MOSS inputs (chunk plan still hybrid for MOSS feasibility).
2. Remap MOSS labels into DiariZen global space (same mapper as Mode A — identity alignment only).
3. Frame/collar voting on speaker activity; prefer mapped MOSS when both agree on speech; else DiariZen; reconstruct turns; attach MOSS text when timing aligns within collar.

## 5. Solving MOSS limits

| Limit | Mitigation |
|-------|------------|
| Duration (~60–90 min) | Hybrid chunker: target 30–60 min, hard cap 90 min; ~30–60 s overlap between adjacent chunks |
| Many speakers (>8) | Prefer windows with ≤8 DiariZen-active speakers; if impossible, still chunk by time and set `high_speaker_density` (lower trust in MOSS IDs, lean DiariZen) |
| Cross-chunk ID drift | DiariZen global centroids + overlap region consistency for `Sxx` remapping |
| Chunk failure / truncate | Keep DiariZen for that span; record `moss_failed` in meta |

## 6. Embeddings

- DiariZen already extracts WeSpeaker embeddings (`pyannote/wespeaker-voxceleb-resnet34-LM`) for clustering.
- Current `DiariZenPipeline.__call__` discards centroids; v1 **exposes** global speaker centroids (and reuses the loaded embedder).
- Do **not** introduce a second unrelated embedding model for DiariZen speakers.
- Extra WeSpeaker forward passes are only for **MOSS turn/speaker crops** needed for cross-system mapping.

## 7. Data schema

### Turn (JSON)

```json
{
  "start": 12.26,
  "end": 13.81,
  "speaker_id": "speaker_2",
  "text": "……",
  "asr_status": "provisional",
  "source": "fused",
  "confidence": 0.87
}
```

- `asr_status`: `provisional` (from MOSS) | `final` (future ASR) | `empty`
- Stage-2 ASR may overwrite `text` and set `asr_status=final` without changing `speaker_id`.

### RTTM

Standard SPEAKER lines only (no text). Used for DER.

### Run meta (in JSON root)

Includes `mode`, DiariZen/MOSS model ids, chunk plan, flags (`high_speaker_density`, `moss_failed` spans), mapping thresholds.

## 8. Components

| Module | Responsibility |
|--------|----------------|
| `fusion_diarize/audio_prep.py` | 16 kHz mono; duration probe |
| `fusion_diarize/diarizen_runner.py` | Load s80-md; return Annotation + centroids; share WeSpeaker for MOSS crops |
| `fusion_diarize/moss_runner.py` | Per-chunk inference; parse segments + text |
| `fusion_diarize/chunk_planner.py` | Hybrid windows + fixed fallback |
| `fusion_diarize/mapper.py` | Overlap + embedding assignment |
| `fusion_diarize/fuse_a.py` | Mode A |
| `fusion_diarize/fuse_b.py` | Mode B |
| `fusion_diarize/export.py` | RTTM + JSON |
| `fusion_diarize/eval_der.py` | DER; baselines and mode comparison |
| `fusion_diarize/cli.py` | `run --mode a\|b\|both`, `eval` |

## 9. CLI (remote experiments)

```bash
# Run both modes, cache shared intermediates
python -m fusion_diarize.cli run \
  --audio /data/utt.wav \
  --work-dir /data/work/utt \
  --mode both

# Evaluate against reference RTTM
python -m fusion_diarize.cli eval \
  --hyp /data/work/utt/mode_a.rttm \
  --ref /data/ref/utt.rttm
```

Batch over a manifest (wav + optional ref RTTM) is in scope for experiment convenience.

## 10. Error handling

- MOSS OOM / truncate / exception on a chunk → DiariZen-only for that time range; non-fatal if any chunk succeeded or DiariZen succeeded.
- No speech → empty RTTM/JSON, success exit.
- Hard failures (missing model, unreadable audio, DiariZen crash) → non-zero exit.

## 11. Evaluation protocol

- **Baselines:** DiariZen-only; MOSS-only (whole file if short enough, else remapped chunk concat).
- **Systems:** Mode A; Mode B.
- **Metric:** DER (optional JER) with standard collar.
- **Slices:** duration buckets; speaker-count buckets (e.g. ≤8 vs >8).
- **Decision:** pick default `--mode` from held-out set; keep CLI switch.

## 12. Testing (engineering)

- Unit tests for chunk planner, mapper assignment, fuse A/B on synthetic turn lists (no GPU).
- Smoke test hooks documented for remote GPU (one short wav).
- Export round-trip: turns → RTTM/JSON → reload.

## 13. Future (explicitly deferred)

- Minimal review UI on top of JSON.
- Dedicated ASR stage filling `text` / `asr_status=final`.
- Stronger fusion (learned DOVER-style weights) if A/B plateaus.
- Cross-session speaker linking.
