# Mode C Fusion Design

**Date:** 2026-07-22  
**Status:** Approved for implementation  
**Goal:** MOSS-primary diarization with DiariZen used only as guardrails (incomplete gaps + extreme speaker-count explosion).

## Background

Experiments showed chunked MOSS alone beats Mode A/B on most files once incompleteness is fixed. Mode A/B inject DiariZen activity into the timeline (OR / gap-fill-too-aggressively), causing false alarm and simultaneous multi-speaker labels. DiariZen speaker count is biased both ways, so it must not trigger “missing speaker” injection via `n_dz vs n_moss`.

## Rules

1. Default output = remapped MOSS boundaries + DiariZen global IDs (no bare `cXXX:Syy` in final output when mapping exists).
2. Never OR DiariZen and MOSS activity on the same frames.
3. DiariZen activity only when:
   - MOSS chunk is incomplete/failed → fill silence gaps only; or
   - MOSS invents an extreme local speaker count → DiariZen backbone + MOSS text attach.
4. Never invent DiariZen-only speakers because `n_dz > n_moss`.

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
