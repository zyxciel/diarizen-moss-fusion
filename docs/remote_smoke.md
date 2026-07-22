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
rm -rf /data/work/utt/chunks.json /data/work/utt/moss_turns.json /data/work/utt/moss

fusion-diarize run \
  --audio /data/utt.wav \
  --work-dir /data/work/utt \
  --mode c \
  --diarizen-repo BUT-FIT/diarizen-wavlm-large-s80-md \
  --moss-model /models/MOSS-Transcribe-Diarize \
  --chunk-max 1200 \
  --force-rechunk

# Mode C RTTM under the work dir (default; MOSS-primary + DiariZen guardrails)
fusion-diarize eval --hyp /data/work/utt/mode_c.rttm --ref /data/ref/utt.rttm
```

Defaults: MOSS chunks ≤ **20 minutes**, `max_new_tokens=65536`, fuse mode **`c`**.

### Mode C behavior

- **Normal:** remapped MOSS only (`meta.fusion_path=moss_primary`) — closest to single MOSS.
- **Incomplete MOSS chunk:** DiariZen fills silence gaps only (`moss_primary_gapfill`).
- **Speaker explosion:** if MOSS local IDs `> max(12, 2 * n_diarizen)`, use DiariZen backbone + attach MOSS text (`diarizen_backbone_explosion`). Check `mode_c.json` → `meta.explosion`.

Ablations: `--mode a`, `--mode b`, or `--mode both` (writes A+B+C).

Eval prints a JSON summary: `der`, `false_alarm`, `missed_detection`, `confusion`, `collar`. Check `mode_c.json` → `meta.n_incomplete_moss_chunks` / `fusion_path` if DER is high.

## 5. Baselines (ablation)

Cached intermediates in `--work-dir` support DiariZen-only and MOSS-only baselines:

- **DiariZen-only:** export `diarizen.json` turns to RTTM (same schema as fused output), then `fusion-diarize eval --hyp … --ref …`
- **MOSS-only:** use remapped MOSS turns (speaker ids aligned to DiariZen) from the pipeline cache / intermediates, write RTTM, then eval

Compare those DERs against `mode_c.rttm` (and optionally `mode_a.rttm` / `mode_b.rttm`).