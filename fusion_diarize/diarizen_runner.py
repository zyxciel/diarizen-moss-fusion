"""DiariZen runner that keeps clustering centroids and shares WeSpeaker embeds.

Path setup (needed to import vendored DiariZen + nested pyannote-audio)::

    workspace/
      DiariZen-main/           # add to PYTHONPATH
      DiariZen-main/pyannote-audio/
      fusion_diarize/

Or rely on this module inserting those two paths relative to the workspace
root (parent of ``fusion_diarize/``) before the lazy DiariZen import.

Does not permanently patch upstream; ``_call_with_centroids`` copies
``DiariZenPipeline.__call__`` but keeps the centroids return value.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from fusion_diarize.audio_prep import load_mono16k
from fusion_diarize.types import Turn

DIARIZEN_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md"

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_DIARIZEN_ROOT = _WORKSPACE_ROOT / "DiariZen-main"
_PYANNOTE_ROOT = _DIARIZEN_ROOT / "pyannote-audio"


def _ensure_diarizen_on_path() -> None:
    """Insert vendored DiariZen and nested pyannote-audio onto ``sys.path``."""
    for root in (_DIARIZEN_ROOT, _PYANNOTE_ROOT):
        s = str(root)
        if root.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


def align_centroids(
    labels: list[str], centroids_array: np.ndarray
) -> dict[str, np.ndarray]:
    """Map speaker labels to centroid rows by list index order.

    ``labels[i]`` corresponds to ``centroids_array[i]``, matching pyannote's
    ``return_embeddings=True`` alignment after label rename.
    """
    out: dict[str, np.ndarray] = {}
    if centroids_array is None:
        return out
    arr = np.asarray(centroids_array)
    for i, lab in enumerate(labels):
        if i < len(arr):
            out[str(lab)] = np.asarray(arr[i], dtype=np.float32)
    return out


def _speaker_label(lab: Any) -> str:
    if isinstance(lab, (int, np.integer)):
        return f"speaker_{int(lab)}"
    s = str(lab)
    if s.startswith("speaker_"):
        return s
    if s.isdigit():
        return f"speaker_{s}"
    return s


class DiariZenRunner:
    """Lazy-loads DiariZen hub model; exposes turns + centroids + crop embeds."""

    def __init__(
        self,
        repo_id: str = DIARIZEN_REPO,
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        _ensure_diarizen_on_path()
        # Heavy: HF snapshot + WeSpeaker ONNX
        from diarizen.pipelines.inference import DiariZenPipeline

        self.pipeline = DiariZenPipeline.from_pretrained(
            repo_id, cache_dir=cache_dir
        )
        if device is not None:
            import torch

            self.pipeline.to(torch.device(device))

    def run(
        self, audio_path: Path, sess_name: str = "utt"
    ) -> tuple[list[Turn], dict[str, np.ndarray]]:
        annotation, centroids = self._call_with_centroids(audio_path, sess_name)
        annotation, cent = self._rename_and_align(annotation, centroids)
        turns: list[Turn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append(
                Turn(float(segment.start), float(segment.end), str(label))
            )
        return turns, cent

    @staticmethod
    def _clean_speech_intervals(
        turn: Turn, all_turns: list[Turn]
    ) -> list[tuple[float, float]]:
        """Return single-speaker intervals for ``turn`` (overlap with others removed)."""
        cuts = [
            (other.start, other.end)
            for other in all_turns
            if other.speaker_id != turn.speaker_id
            and other.end > turn.start
            and other.start < turn.end
        ]
        if not cuts:
            return [(turn.start, turn.end)] if turn.end > turn.start else []
        cuts.sort()
        merged: list[list[float]] = [[cuts[0][0], cuts[0][1]]]
        for start, end in cuts[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        clean: list[tuple[float, float]] = []
        cursor = turn.start
        for cut_start, cut_end in merged:
            if cut_end <= cursor:
                continue
            if cut_start >= turn.end:
                break
            piece_end = min(cut_start, turn.end)
            if piece_end - cursor > 1e-3:
                clean.append((cursor, piece_end))
            cursor = max(cursor, cut_end)
            if cursor >= turn.end:
                break
        if turn.end - cursor > 1e-3:
            clean.append((cursor, turn.end))
        return clean

    def embed_moss_labels(
        self, audio_path: Path, moss_turns: list[Turn]
    ) -> dict[str, np.ndarray]:
        """Duration-weight WeSpeaker embeddings over **non-overlapping** crops.

        Overlapping multi-speaker regions are excluded so pooled embeddings stay
        speaker-pure. Skips zero/nonfinite vectors. Returns {} when empty.
        """
        if not moss_turns:
            return {}
        wav, sr = load_mono16k(audio_path)
        assert sr == 16000
        by_label: dict[str, list[np.ndarray]] = {}
        durations: dict[str, list[float]] = {}
        for t in moss_turns:
            for start, end in self._clean_speech_intervals(t, moss_turns):
                emb = self._embed_region(wav, start, end)
                if np.all(np.isfinite(emb)) and np.any(emb != 0):
                    by_label.setdefault(t.speaker_id, []).append(emb)
                    i0 = max(0, int(round(start * sr)))
                    i1 = min(len(wav), int(round(end * sr)))
                    durations.setdefault(t.speaker_id, []).append(
                        max((i1 - i0) / sr, 1e-3)
                    )
        return {
            k: np.average(
                np.stack(v, axis=0), axis=0, weights=durations[k]
            ).astype(np.float32)
            for k, v in by_label.items()
        }

    def embed_crop(
        self, waveform_16k: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """Embed a time region of a mono 16 kHz waveform via pipeline WeSpeaker."""
        return self._embed_region(waveform_16k, start, end)

    def _embed_region(
        self, wav: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """Call ``pipeline._embedding`` as ``(batch, channel, time)`` tensor.

        Matches ``get_embeddings`` in pyannote ``speaker_diarization.py``:
        ``self._embedding(waveform_batch, masks=...)`` with
        ``waveform_batch`` shaped ``(batch, 1, num_samples)``.
        Short crops return a zero vector of embedding dimension.
        """
        import torch

        emb_model = self.pipeline._embedding
        dim = int(emb_model.dimension)
        sr = int(getattr(emb_model, "sample_rate", 16000))
        i0 = max(0, int(round(start * sr)))
        i1 = min(len(wav), int(round(end * sr)))
        if i1 <= i0:
            return np.zeros(dim, dtype=np.float32)

        crop = np.asarray(wav[i0:i1], dtype=np.float32)
        min_samples = int(getattr(emb_model, "min_num_samples", 1))
        if crop.shape[0] < min_samples:
            return np.zeros(dim, dtype=np.float32)

        # (batch, channel, time) — same layout as get_embeddings
        waveform_batch = torch.from_numpy(crop)[None, None, :]
        out = emb_model(waveform_batch)
        vec = np.asarray(out[0], dtype=np.float32)
        if vec.shape[0] != dim or np.any(~np.isfinite(vec)):
            return np.zeros(dim, dtype=np.float32)
        return vec

    def _rename_and_align(
        self, annotation, centroids: np.ndarray | None
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Rename int labels to ``speaker_k`` and align centroid rows."""
        orig_labels = list(annotation.labels())
        mapping = {lab: _speaker_label(lab) for lab in orig_labels}
        annotation = annotation.rename_labels(mapping=mapping)
        new_labels = [str(l) for l in annotation.labels()]

        if centroids is None:
            return annotation, {}

        centroids = np.asarray(centroids)
        # Pad if annotation has more speakers than centroid rows (pyannote style)
        if len(new_labels) > centroids.shape[0]:
            centroids = np.pad(
                centroids,
                ((0, len(new_labels) - centroids.shape[0]), (0, 0)),
            )

        # Re-order so rows match annotation.labels() order
        inverse = {new: old for old, new in mapping.items()}
        row_indices: list[int] = []
        for lab in new_labels:
            old = inverse[lab]
            if isinstance(old, (int, np.integer)):
                row_indices.append(int(old))
            elif str(old).isdigit():
                row_indices.append(int(old))
            else:
                row_indices.append(orig_labels.index(old))

        max_idx = max(row_indices) if row_indices else -1
        if max_idx >= centroids.shape[0]:
            centroids = np.pad(
                centroids, ((0, max_idx + 1 - centroids.shape[0]), (0, 0))
            )
        centroids = centroids[row_indices]
        return annotation, align_centroids(new_labels, centroids)

    def _call_with_centroids(self, audio_path, sess_name: str | None = None):
        """Copy of ``DiariZenPipeline.__call__`` keeping clustering centroids.

        Upstream discards centroids at::

            hard_clusters, _, _ = self.clustering(...)

        Here we keep the third return value.
        """
        import os

        import torch
        import torchaudio
        from scipy.ndimage import median_filter
        from pyannote.database.protocol.protocol import ProtocolFile

        pipeline = self.pipeline
        in_wav = audio_path if not isinstance(audio_path, Path) else str(audio_path)

        assert isinstance(in_wav, (str, BytesIO, ProtocolFile)), (
            f"input must be either a str, BytesIO or a ProtocolFile; "
            f"there was {type(in_wav)}"
        )
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav["audio"]

        print("Extracting segmentations.")
        waveform, sample_rate = torchaudio.load(in_wav)
        waveform = torch.unsqueeze(waveform[0], 0)  # force SDM / mono
        segmentations = pipeline.get_segmentations(
            {"waveform": waveform, "sample_rate": sample_rate}, soft=False
        )

        if pipeline.apply_median_filtering:
            segmentations.data = median_filter(
                segmentations.data, size=(1, 11, 1), mode="reflect"
            )

        binarized_segmentations = segmentations  # powerset

        count = pipeline.speaker_count(
            binarized_segmentations,
            pipeline._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
        embeddings = pipeline.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=pipeline.embedding_exclude_overlap,
        )

        print("Clustering.")
        hard_clusters, _, centroids = pipeline.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=pipeline.min_speakers,
            max_clusters=pipeline.max_speakers,
        )

        count.data = np.minimum(count.data, pipeline.max_speakers).astype(np.int8)

        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2

        # `reconstruct()` calls `to_diarization()` which returns
        # (discrete_diarization, activations) — we only need the first.
        discrete_diarization, _ = pipeline.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # Use pipeline.to_annotation so rename_tracks(generator="string") is
        # applied, matching what pyannote's apply() does (integer labels → "0",
        # "1", …).  A bare Binarize() call leaves integer track labels which
        # break downstream itertracks assumptions.
        result = pipeline.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=0.0,
        )
        result.uri = sess_name

        if pipeline.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(pipeline.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())

        return result, centroids


class FakeDiariZenRunner:
    """Offline stand-in for pipeline tests (no HF / GPU)."""

    def __init__(
        self,
        turns: list[Turn],
        centroids: dict[str, np.ndarray] | None = None,
    ):
        self.turns = list(turns)
        self.centroids = dict(centroids) if centroids is not None else {}

    def run(
        self, path: Path, sess_name: str = "utt"
    ) -> tuple[list[Turn], dict[str, np.ndarray]]:
        return list(self.turns), dict(self.centroids)

    def embed_moss_labels(
        self, path: Path, moss_turns: list[Turn]
    ) -> dict[str, np.ndarray]:
        labels = sorted({t.speaker_id for t in moss_turns})
        return {lab: np.ones(2, dtype=np.float32) for lab in labels}
