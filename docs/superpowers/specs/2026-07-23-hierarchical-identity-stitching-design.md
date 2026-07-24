# Hierarchical MOSS Speaker Identity Stitching Design

**Date:** 2026-07-23  
**Status:** Approved design, pending implementation plan  
**Goal:** Reduce cross-chunk speaker confusion by assigning stable global identities to chunk-local MOSS speakers without allowing DiariZen speaker-count errors to override stronger MOSS evidence.

## 1. Background

Mode C uses MOSS as the primary diarization timeline and DiariZen only for incomplete gaps and extreme failures. Experiments show that its largest remaining error is confusion: the same person receives different speaker IDs in adjacent chunks.

The existing mapper independently scores every namespaced MOSS label (`cXXX:SYY`) against DiariZen global speakers using aggregate temporal overlap and embedding cosine, then performs greedy assignment. It does not explicitly match the two MOSS predictions in adjacent chunk overlaps. Consequently, DiariZen splits, merges, or weak greedy assignments can cause cross-chunk identity drift.

The replacement is an upstream identity-stitching layer. This is not a new fusion Mode D: Mode C still decides which system supplies timeline activity, while the stitcher decides which global identity each MOSS local speaker represents.

## 2. Locked decisions

1. MOSS-to-MOSS agreement in adjacent chunk overlaps is the strongest identity evidence.
2. DiariZen temporal overlap provides a global anchor, not an unconditional merge instruction.
3. Speaker embeddings link speakers absent from overlaps and resolve ambiguous mappings.
4. Two different MOSS speakers from the same chunk are an absolute cannot-link pair.
5. Uncertain non-overlap matches remain separate rather than being forcibly merged.
6. Cluster transitivity is bounded by embedding-diameter checks to prevent chaining.
7. Final IDs represent stitched MOSS clusters. DiariZen IDs are stored as supporting anchors.
8. Chunk-overlap predictions are reconciled so only one chunk supplies activity at each frame; genuine multi-speaker overlap inside that selected chunk is retained.
9. Existing Turn JSON and RTTM schemas remain unchanged.

## 3. Architecture

```text
Audio
 ├─ DiariZen full-file ───────────────────────────────┐
 └─ MOSS chunk inference → local speaker nodes        │
                              ↓                       │
                  HierarchicalIdentityStitcher ←──────┘
                   ├─ adjacent-overlap links
                   ├─ MOSS embedding links
                   ├─ DiariZen temporal anchors
                   ├─ cannot-link constraints
                   ├─ diameter validation
                   └─ overlap ownership reconciliation
                              ↓
                    globally stitched MOSS turns
                              ↓
                  Mode C system-exclusive fusion
                              ↓
                         RTTM + JSON
```

`fusion_diarize/identity_stitcher.py` owns identity construction and overlap reconciliation. It does not select MOSS versus DiariZen activity outside chunk-overlap ownership. The legacy mapper remains available for reproducible Mode A/B experiments.

## 4. Local speaker nodes

Each chunk-local MOSS speaker is represented by:

- namespaced local ID and chunk index;
- global-time turns and duration;
- activity restricted to each neighboring chunk overlap;
- a duration-weighted mean WeSpeaker embedding from multiple clean speech crops;
- temporal-overlap distribution over DiariZen global speakers;
- accepted DiariZen anchor and its confidence, when available.

Zero-norm or unavailable embeddings are marked missing and never treated as evidence.

## 5. Adjacent-overlap matching

For adjacent chunks, restrict each candidate pair to the shared overlap window \(W\). Quantize activity at a 20 ms frame hop:

\[
A_t = 1 \text{ when local speaker } a \text{ is active at frame } t
\]

\[
B_t = 1 \text{ when local speaker } b \text{ is active at frame } t
\]

Temporal activity similarity is frame IoU:

\[
\operatorname{IoU}(a,b)=
\frac{\sum_{t\in W} A_tB_t}
{\sum_{t\in W}\max(A_t,B_t)}
\]

A pair is eligible only when:

- both speakers have at least 2.0 seconds of activity in the overlap;
- intersection duration is at least 1.0 second;
- the IoU denominator is nonzero.

The pair score is:

\[
S(a,b)=0.7\operatorname{IoU}(a,b)
       +0.3\operatorname{CosineNorm}(e_a,e_b)
\]

where normalized cosine is \((1+\cos(e_a,e_b))/2\). When an embedding is missing, temporal evidence is used alone and the diagnostic records the missing component.

Concretely, when either embedding is missing, set \(S(a,b)=\operatorname{IoU}(a,b)\); do not multiply IoU by 0.7.

Use Hungarian one-to-one assignment within each adjacent chunk overlap. Accept a link only when:

- combined score is at least 0.70; and
- its margin over the next candidate for either endpoint is at least 0.10.

The 20 ms IoU is exact by default because both chunks refer to the same source timeline. A 100 ms boundary collar may be exposed for experiments but defaults to zero.

## 6. DiariZen anchors

For MOSS local speaker \(m\), compute overlap duration with each DiariZen speaker \(d\):

\[
\operatorname{purity}(m)=
\frac{\operatorname{overlap}(m,d_{\mathrm{best}})}
{\sum_d\operatorname{overlap}(m,d)}
\]

Accept the best DiariZen anchor only when:

- best overlap is at least 3.0 seconds;
- purity is at least 0.70;
- an available MOSS-to-DiariZen embedding cosine is at least 0.40. A lower cosine explicitly contradicts and rejects the anchor.

DiariZen anchors are weak merge evidence. Two MOSS clusters sharing one DiariZen anchor are not merged automatically because DiariZen under-counts speakers on high-speaker recordings. Strong adjacent MOSS overlap overrides a conflicting DiariZen anchor.

## 7. Non-adjacent embedding links

Speakers absent from adjacent overlaps may link through their pooled MOSS embeddings. A candidate requires cosine similarity of at least 0.80 and a margin of at least 0.05 over the next candidate for either endpoint. When cosine candidates are within 0.03, compatible DiariZen anchors select the tie-break winner. A conflicting DiariZen anchor raises the required cosine to 0.85 but does not reject a match above that threshold.

Embedding-only candidates use a stricter acceptance policy than overlap links. If the best candidate is not clearly separated from alternatives, keep the local speaker in a separate cluster.

## 8. Constrained clustering and chaining protection

Represent accepted relationships as weighted graph edges. Process candidates from strongest to weakest. Before joining two connected components:

1. reject the merge if it would put two different local speakers from the same chunk into one cluster;
2. compute the proposed cluster's maximum pairwise cosine distance:

\[
D(C)=\max_{i,j\in C}(1-\cos(e_i,e_j))
\]

3. reject the merge when \(D(C)>0.30\), ignoring missing-embedding pairs rather than treating them as matches.

After all merges, validate every cluster. While a cluster exceeds the diameter threshold, remove its weakest accepted edge and recompute connected components. Record:

- maximum pairwise distance;
- maximum distance to the cluster medoid;
- removed edge and split reason.

This prevents the union-find chaining trap in long recordings, where individually plausible A–B and B–C links could incorrectly imply A–C identity.

## 9. Final IDs and diagnostics

Generate final IDs independently as `speaker_0`, `speaker_1`, and so on, ordering clusters by their earliest activity time and then by local ID for deterministic output. Store DiariZen association separately:

```json
{
  "c001:S03": {
    "global_id": "speaker_0",
    "method": "moss_overlap",
    "confidence": 0.94,
    "diarizen_anchor": "speaker_3",
    "diarizen_purity": 0.82
  }
}
```

Identity metadata also includes accepted and rejected edges, temporal IoU, embedding cosine, assignment margin, cluster diameter, conflicts, and configured thresholds.

## 10. Chunk-overlap output reconciliation

The chunk overlap is used for matching, but two chunk predictions must not both be exported. Assign one chunk as owner at every overlap frame:

- earlier chunk owns the left half;
- later chunk owns the right half;
- ownership switches at the overlap midpoint.

Clip turns to their owned spans, then merge adjacent same-speaker turns. This removes duplicate chunk predictions while preserving true simultaneous speakers produced within the owning chunk.

## 11. Mode C integration and fallback

Pipeline order:

1. run chunked MOSS;
2. detect per-chunk MOSS speaker explosion;
3. stitch identities and reconcile overlap ownership;
4. run Mode C with stitched MOSS turns;
5. export output and identity diagnostics.

The current whole-recording explosion count is incorrect because namespaced local IDs accumulate across chunks. Detect explosion per chunk, counting DiariZen speakers that have activity inside that chunk:

\[
n_{\mathrm{MOSS,chunk}} > 12
\quad\land\quad
n_{\mathrm{MOSS,chunk}} >
2\max(n_{\mathrm{DiariZen,chunk}},1)
\]

An exploded chunk uses DiariZen only on that chunk's owned span. Normal chunks remain MOSS-primary. After stitching, global speaker count is diagnostic and never triggers timeline replacement.

Other failure behavior:

- missing overlap link: try embedding linkage;
- uncertain embedding link: keep separate ID;
- missing DiariZen output: continue with MOSS overlap and embeddings;
- incomplete MOSS chunk: DiariZen fills only MOSS-silent regions in that span;
- all MOSS chunks fail: full DiariZen fallback.

## 12. Compatibility and caching

- Turn JSON fields and RTTM format do not change.
- Add root metadata under `identity_stitching`.
- Add CLI strategy `--identity-map hierarchical|legacy`.
- Mode C defaults to `hierarchical`.
- Mode A/B retain legacy mapping for experiment reproducibility.
- Cache stitched mappings separately and include an algorithm/version key plus thresholds. Threshold or algorithm changes invalidate identity mappings without forcing MOSS inference to rerun.

## 13. Initial thresholds

| Parameter | Default |
|---|---:|
| Frame hop | 0.02 s |
| Minimum overlap activity per endpoint | 2.0 s |
| Minimum intersection duration | 1.0 s |
| Adjacent overlap link score | 0.70 |
| Assignment margin | 0.10 |
| Non-adjacent embedding cosine | 0.80 |
| Non-adjacent embedding margin | 0.05 |
| Conflicting-anchor embedding cosine | 0.85 |
| Cluster maximum cosine distance | 0.30 |
| DiariZen anchor purity | 0.70 |
| DiariZen minimum overlap | 3.0 s |
| DiariZen anchor minimum cosine | 0.40 |

Store defaults in metadata and tune them on a development subset, not the final test set.

## 14. Testing

Synthetic tests cover:

1. adjacent chunks with different local IDs for the same speaker;
2. simultaneous speakers remaining distinct;
3. same-chunk cannot-link enforcement;
4. absent-overlap speaker linked by strong embedding;
5. weak non-overlap match kept separate;
6. strong MOSS overlap overriding a conflicting DiariZen anchor;
7. distinct MOSS clusters sharing a DiariZen anchor;
8. A–B–C chaining split by diameter validation;
9. overlap ownership producing no duplicate exported intervals;
10. per-chunk explosion causing span-local DiariZen fallback;
11. incomplete-chunk system-exclusive gap fill;
12. missing embeddings and missing DiariZen graceful fallback.

Evaluation compares legacy Mode C, hierarchical Mode C, chunked MOSS-only, and DiariZen-only on identical files. Report:

- DER, false alarm, miss, and confusion;
- cross-chunk identity-switch count;
- final speaker count;
- uncertain/unlinked local count;
- diameter-triggered cluster splits;
- percentage of audio replaced or filled by DiariZen.

Primary acceptance criterion: hierarchical Mode C reduces both confusion and cross-chunk identity-switch count relative to legacy Mode C, while neither false alarm nor miss increases by more than 0.5 absolute DER percentage points relative to chunked MOSS-only on the designated development set.

## 15. Out of scope

- Cross-recording speaker identity;
- supervised speaker enrollment;
- retraining MOSS or DiariZen;
- automatic threshold tuning on the final evaluation set;
- changing the Turn or RTTM schema.
