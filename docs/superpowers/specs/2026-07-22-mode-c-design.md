# Mode C Fusion Design

**Date:** 2026-07-22  
**Status:** Approved for implementation  
**Goal:** MOSS-primary diarization with DiariZen used only as guardrails (incomplete gaps + extreme speaker-count explosion).

## Background

Experiments showed chunked MOSS alone beats Mode A/B on most files once incompleteness is fixed. Mode A/B inject DiariZen activity into the timeline (OR / gap-fill-too-aggressively), causing false alarm and simultaneous multi-speaker labels. DiariZen speaker count is biased both ways, so it must not trigger “missing speaker” injection via `n_dz vs n_moss`.

## Rules

1. Default output = remapped MOSS boundaries + DiariZen global IDs (no bare `cXXX:Syy` in final output when mapping exists).
2. **System exclusivity:** on any time region, labels come from **one** system only. MOSS always wins where it has speech; DiariZen must not co-label that time (even with a different speaker id).
3. True multi-speaker overlap **within MOSS** is kept.
4. DiariZen activity only when:
   - MOSS chunk is incomplete/failed → fill silence **inside those chunk spans only**, and only where MOSS has no speech; or
   - MOSS invents an extreme local speaker count → DiariZen becomes the **backbone timeline** (single-system path).
5. Never invent DiariZen-only speakers because `n_dz > n_moss`.

## Gates

| Condition | Path | `fusion_path` |
|-----------|------|----------------|
| `n_moss_local > max(12, 2 * n_dz)` | DiariZen turns + attach MOSS text (collar) | `diarizen_backbone_explosion` |
| Else if any incomplete MOSS chunk | Remapped MOSS + DiariZen silent-gap fill | `moss_primary_gapfill` |
| Else | Remapped MOSS only (deduped) | `moss_primary` |

## Out of scope (v1)

- Shorter re-MOSS for true N>6 under-count
- Count-based speaker invention from DiariZen
- Changing 20 min / 65536 token defaults
