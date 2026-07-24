# Remote GPU smoke test

Commands to copy this workspace to a GPU server, install dependencies, and run a short fusion-diarize smoke + DER eval.

## 1. Copy workspace to the remote GPU server

From your laptop (adjust host/path):

```bash
rsync -avz --progress \
  /path/to/diarizen+moss/ \
  user@gpu-server:/data/diarizen+moss/
```

Or with `scp -r`. Keep `DiariZen-main/`, `MOSS-Transcribe-Diarize-main/`, and the `fusion_diarize/` package together.

## 2. Install DiariZen and MOSS dependencies

On the server, follow the upstream install docs (CUDA / torch versions matter):

- **DiariZen:** see [`DiariZen-main/README.md`](../DiariZen-main/README.md) (conda env, `pip install -r requirements.txt && pip install -e .`, then `pyannote-audio`)
- **MOSS-Transcribe-Diarize:** see [`MOSS-Transcribe-Diarize-main/README.md`](../MOSS-Transcribe-Diarize-main/README.md) (environment setup + model download)

`pyannote.metrics` (from DiariZen / pyannote stack) is preferred for DER; `fusion_diarize.eval_der` falls back to a frame-based DER if pyannote is unavailable.

## 3. Install this package (workspace root)

```bash
cd /data/diarizen+moss
pip install -e ".[dev]"
```

## 4. Example fusion-diarize run + eval (default `--mode c`)

**Models**

| Component | Hugging Face id / path |
|-----------|------------------------|
| DiariZen | `BUT-FIT/diarizen-wavlm-large-s80-md` |
| MOSS | `OpenMOSS-Team/MOSS-Transcribe-Diarize` or a local checkpoint path |

```bash
# IMPORTANT: after pulling fusion fixes, delete old work-dir caches so chunks
# are re-planned at 20 min (stale 30–60 min chunks cause MOSS truncation).
# Also drop identity + cache_manifest when chunking or identity thresholds change.
rm -rf /data/work/utt/chunks.json /data/work/utt/moss_turns.json \
  /data/work/utt/moss /data/work/utt/identity_stitching.json \
  /data/work/utt/cache_manifest.json

fusion-diarize run \
  --audio /data/utt.wav \
  --work-dir /data/work/utt \
  --mode c \
  --identity-map hierarchical \
  --diarizen-repo BUT-FIT/diarizen-wavlm-large-s80-md \
  --moss-model /models/MOSS-Transcribe-Diarize \
  --chunk-max 1200 \
  --force-rechunk

# Mode C RTTM under the work dir (default; MOSS-primary + DiariZen guardrails)
fusion-diarize eval --hyp /data/work/utt/mode_c.rttm --ref /data/ref/utt.rttm
```

Defaults: MOSS chunks ≤ **20 minutes**, `max_new_tokens=65536`, fuse mode **`c`**, identity map **`hierarchical`**.

### Mode C behavior

- **Normal:** stitched MOSS only (`meta.fusion_path=moss_primary`) — closest to single MOSS.
- **Incomplete MOSS chunk:** DiariZen fills silence gaps only (`moss_primary_gapfill`).
- **Speaker explosion:** per-chunk owned spans with too many local IDs fall back to DiariZen in those spans only (`moss_primary_with_explosion_fallback`). Check `mode_c.json` → `meta.exploded_spans`.

### Identity stitching (Mode C default)

`--identity-map hierarchical` (default) runs the upstream hierarchical stitcher before Mode C fusion. Ablation: `--identity-map legacy` keeps the old DiariZen-greedy mapper for Mode C.

Cached / diagnostic artifacts:

| Artifact | Role |
|----------|------|
| `identity_stitching.json` | Versioned stitched turns + mapping + stitch diagnostics |
| `cache_manifest.json` | Audio / model / chunk-parameter fingerprints; mismatch invalidates dependents |
| `mode_c.json` → `meta.identity_stitching` | Accepted/rejected edges, `exploded_chunks`, anchors |
| `mode_c.json` → `meta.identity_stitching.cross_chunk_identity_switch_count` | Direct cross-chunk fragmentation diagnostic |

When tuning identity thresholds in development, delete `identity_stitching.json` (or pass `--force-rechunk`) so the stitcher recomputes. Changing audio, DiariZen/MOSS model ids, or chunk parameters updates `cache_manifest.json` and cascades invalidation automatically.

Ablations: `--mode a`, `--mode b`, or `--mode both` (writes A+B+C; A/B still use the legacy mapper).

Eval prints a JSON summary: `der`, `false_alarm`, `missed_detection`, `confusion`, `collar`. Check `mode_c.json` → `meta.n_incomplete_moss_chunks` / `fusion_path` / `identity_stitching.cross_chunk_identity_switch_count` if DER is high.

## 5. Baselines (ablation)

Cached intermediates in `--work-dir` support DiariZen-only and MOSS-only baselines:

- **DiariZen-only:** export `diarizen.json` turns to RTTM (same schema as fused output), then `fusion-diarize eval --hyp … --ref …`
- **MOSS-only:** use remapped MOSS turns (speaker ids aligned to DiariZen) from the pipeline cache / intermediates, write RTTM, then eval

Compare those DERs against `mode_c.rttm` (and optionally `mode_a.rttm` / `mode_b.rttm`).