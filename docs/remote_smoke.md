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

## 4. Example fusion-diarize run + eval (`--mode both`)

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
  --mode a \
  --diarizen-repo BUT-FIT/diarizen-wavlm-large-s80-md \
  --moss-model /models/MOSS-Transcribe-Diarize \
  --chunk-max 1200 \
  --force-rechunk

# Mode A RTTM under the work dir (preferred experimentally)
fusion-diarize eval --hyp /data/work/utt/mode_a.rttm --ref /data/ref/utt.rttm
```

Defaults: MOSS chunks ≤ **20 minutes**, `max_new_tokens=65536` (MOSS long-audio recommendation). Mode A fills DiariZen into MOSS gaps and drops dual hypotheses that caused high FA.

Eval prints a JSON summary: `der`, `false_alarm`, `missed_detection`, `confusion`, `collar`. Check `mode_a.json` → `meta.n_incomplete_moss_chunks` if miss is still high.

## 5. Baselines (ablation)

Cached intermediates in `--work-dir` support DiariZen-only and MOSS-only baselines:

- **DiariZen-only:** export `diarizen.json` turns to RTTM (same schema as fused output), then `fusion-diarize eval --hyp … --ref …`
- **MOSS-only:** use remapped MOSS turns (speaker ids aligned to DiariZen) from the pipeline cache / intermediates, write RTTM, then eval

Compare those DERs against `mode_a.rttm` / `mode_b.rttm` from `--mode both`.
