"""Diarization Error Rate (DER) evaluation vs reference RTTM."""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from fusion_diarize.export import read_rttm_as_turns
from fusion_diarize.types import Turn

_FRAME_HOP = 0.01


def compute_der(
    hyp_rttm: str | Path,
    ref_rttm: str | Path,
    collar: float = 0.25,
) -> dict[str, Any]:
    """Compute DER between hypothesis and reference RTTM files.

    Prefers ``pyannote.metrics`` when importable; otherwise uses a frame-based
    fallback with Hungarian speaker mapping on 10 ms grids.
    """
    hyp_path = Path(hyp_rttm)
    ref_path = Path(ref_rttm)

    try:
        return _compute_der_pyannote(hyp_path, ref_path, collar=collar)
    except ImportError:
        return _compute_der_fallback(hyp_path, ref_path, collar=collar)


def _compute_der_pyannote(
    hyp_path: Path, ref_path: Path, collar: float
) -> dict[str, Any]:
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_ann = _first_annotation(load_rttm(str(ref_path)))
    hyp_ann = _first_annotation(load_rttm(str(hyp_path)))

    metric = DiarizationErrorRate(collar=collar)
    components = metric.compute_components(ref_ann, hyp_ann)
    der = float(metric.compute_metric(components))

    total = float(components.get("total", 0.0)) or 1.0
    fa = float(components.get("false alarm", 0.0))
    miss = float(components.get("missed detection", 0.0))
    conf = float(components.get("confusion", 0.0))

    return {
        "der": der,
        "false_alarm": fa / total,
        "missed_detection": miss / total,
        "confusion": conf / total,
        "collar": collar,
        "speaker_mapped": True,
        "backend": "pyannote",
    }


def _first_annotation(loaded: dict) -> Any:
    if not loaded:
        raise ValueError("RTTM produced an empty annotation")
    return next(iter(loaded.values()))


def _compute_der_fallback(
    hyp_path: Path, ref_path: Path, collar: float
) -> dict[str, Any]:
    """Frame DER with Hungarian speaker mapping (no pyannote)."""
    hyp_turns = read_rttm_as_turns(hyp_path)
    ref_turns = read_rttm_as_turns(ref_path)

    if not ref_turns and not hyp_turns:
        return {
            "der": 0.0,
            "false_alarm": 0.0,
            "missed_detection": 0.0,
            "confusion": 0.0,
            "collar": collar,
            "speaker_mapped": True,
            "backend": "fallback",
        }

    t_max = max(
        [0.0]
        + [t.end for t in ref_turns]
        + [t.end for t in hyp_turns]
    )
    n_frames = max(1, int(np.ceil(t_max / _FRAME_HOP)))

    ref_labels = sorted({t.speaker_id for t in ref_turns})
    hyp_labels = sorted({t.speaker_id for t in hyp_turns})

    ref_mat = _activity_matrix(ref_turns, ref_labels, n_frames)
    hyp_mat = _activity_matrix(hyp_turns, hyp_labels, n_frames)

    mapping = _hungarian_map_by_overlap(ref_mat, hyp_mat, ref_labels, hyp_labels)
    hyp_mapped = _remap_hyp_matrix(hyp_mat, hyp_labels, ref_labels, mapping)

    scored = _collar_mask(ref_turns, n_frames, collar)

    fa_f, miss_f, conf_f, total_f = _frame_components(
        ref_mat, hyp_mapped, scored
    )
    denom = total_f if total_f > 0 else 1.0
    der = (fa_f + miss_f + conf_f) / denom

    return {
        "der": float(der),
        "false_alarm": float(fa_f / denom),
        "missed_detection": float(miss_f / denom),
        "confusion": float(conf_f / denom),
        "collar": collar,
        "speaker_mapped": True,
        "backend": "fallback",
    }


def _activity_matrix(
    turns: list[Turn], labels: list[str], n_frames: int
) -> np.ndarray:
    mat = np.zeros((len(labels), n_frames), dtype=bool)
    if not labels:
        return mat
    idx = {s: i for i, s in enumerate(labels)}
    for t in turns:
        i = idx.get(t.speaker_id)
        if i is None:
            continue
        a = max(0, int(t.start / _FRAME_HOP))
        b = min(n_frames, int(np.ceil(t.end / _FRAME_HOP)))
        if b > a:
            mat[i, a:b] = True
    return mat


def _hungarian_map_by_overlap(
    ref_mat: np.ndarray,
    hyp_mat: np.ndarray,
    ref_labels: list[str],
    hyp_labels: list[str],
) -> dict[str, str]:
    """Map hyp speaker -> ref speaker maximizing overlap (Hungarian / brute)."""
    n_ref, n_hyp = len(ref_labels), len(hyp_labels)
    if n_ref == 0 or n_hyp == 0:
        return {}

    # ref_mat (n_ref, T) @ hyp_mat.T (T, n_hyp) → overlap (n_ref, n_hyp)
    overlap = ref_mat.astype(np.int64) @ hyp_mat.astype(np.int64).T

    try:
        from scipy.optimize import linear_sum_assignment

        ri, hi = linear_sum_assignment(-overlap)
        return {
            hyp_labels[j]: ref_labels[i]
            for i, j in zip(ri, hi)
            if i < n_ref and j < n_hyp
        }
    except ImportError:
        pass

    # Brute-force assignment for small speaker sets
    n = max(n_ref, n_hyp)
    best_score = -1.0
    best: dict[str, str] = {}
    for ref_perm in permutations(range(n), n_hyp):
        score = 0.0
        cand: dict[str, str] = {}
        for j, ri in enumerate(ref_perm):
            if ri >= n_ref:
                continue
            score += float(overlap[ri, j])
            cand[hyp_labels[j]] = ref_labels[ri]
        if score > best_score:
            best_score = score
            best = cand
    return best


def _remap_hyp_matrix(
    hyp_mat: np.ndarray,
    hyp_labels: list[str],
    ref_labels: list[str],
    mapping: dict[str, str],
) -> np.ndarray:
    """Project hyp activity into ref label rows; unmapped hyp kept as extra rows."""
    n_ref = len(ref_labels)
    n_frames = int(hyp_mat.shape[1]) if hyp_mat.ndim == 2 else 0
    out = np.zeros((n_ref, n_frames), dtype=bool)
    if n_frames == 0:
        return out
    ref_idx = {s: i for i, s in enumerate(ref_labels)}
    unmapped_rows: list[np.ndarray] = []

    for j, hlab in enumerate(hyp_labels):
        rlab = mapping.get(hlab)
        if rlab is not None and rlab in ref_idx:
            out[ref_idx[rlab]] |= hyp_mat[j]
        else:
            unmapped_rows.append(hyp_mat[j])

    if unmapped_rows:
        extra = np.stack(unmapped_rows, axis=0)
        out = np.concatenate([out, extra], axis=0)
    return out


def _collar_mask(ref_turns: list[Turn], n_frames: int, collar: float) -> np.ndarray:
    """True where frames are scored; False near reference boundaries (collar)."""
    scored = np.ones(n_frames, dtype=bool)
    if collar <= 0 or not ref_turns:
        return scored
    c = int(round(collar / _FRAME_HOP))
    for t in ref_turns:
        a = max(0, int(t.start / _FRAME_HOP))
        b = min(n_frames, int(np.ceil(t.end / _FRAME_HOP)))
        scored[max(0, a - c) : min(n_frames, a + c)] = False
        scored[max(0, b - c) : min(n_frames, b + c)] = False
    return scored


def _frame_components(
    ref_mat: np.ndarray,
    hyp_mapped: np.ndarray,
    scored: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return FA, MISS, CONF, TOTAL as frame counts (speaker-frames)."""
    n_ref = ref_mat.shape[0]
    # Align hyp rows: first n_ref are mapped; extras are unmapped (FA only)
    if hyp_mapped.shape[0] < n_ref:
        pad = np.zeros(
            (n_ref - hyp_mapped.shape[0], hyp_mapped.shape[1]), dtype=bool
        )
        hyp_mapped = np.concatenate([hyp_mapped, pad], axis=0)

    hyp_ref = hyp_mapped[:n_ref]
    hyp_extra = hyp_mapped[n_ref:] if hyp_mapped.shape[0] > n_ref else None

    fa = miss = conf = total = 0.0
    n_frames = ref_mat.shape[1]

    for t in range(n_frames):
        if not scored[t]:
            continue
        R = set(np.where(ref_mat[:, t])[0].tolist()) if n_ref else set()
        H = set(np.where(hyp_ref[:, t])[0].tolist()) if n_ref else set()
        if hyp_extra is not None and hyp_extra[:, t].any():
            # treat each unmapped active row as a distinct FA speaker
            for k in np.where(hyp_extra[:, t])[0]:
                H.add(n_ref + int(k))

        if not R and not H:
            continue
        if not R:
            fa += len(H)
            continue

        total += len(R)
        only_r = R - H
        only_h = H - R
        n_conf = min(len(only_r), len(only_h))
        conf += n_conf
        miss += len(only_r) - n_conf
        fa += len(only_h) - n_conf

    return fa, miss, conf, total
