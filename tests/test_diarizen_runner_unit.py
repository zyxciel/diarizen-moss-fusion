"""Unit tests for DiariZen runner helpers (no HF / GPU / from_pretrained)."""
import numpy as np


def test_align_centroids_to_labels():
    from fusion_diarize.diarizen_runner import align_centroids

    labels = ["speaker_1", "speaker_0"]
    cents = np.array([[0.0, 1.0], [1.0, 0.0]])
    d = align_centroids(labels, cents)
    assert set(d) == {"speaker_0", "speaker_1"}
    np.testing.assert_array_equal(d["speaker_1"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(d["speaker_0"], np.array([1.0, 0.0], dtype=np.float32))


def test_diarizen_runner_module_imports_without_constructing():
    """Importing the module must not load DiariZenPipeline / HF models."""
    import fusion_diarize.diarizen_runner as mod

    assert hasattr(mod, "DiariZenRunner")
    assert hasattr(mod, "align_centroids")
    assert mod.DIARIZEN_REPO == "BUT-FIT/diarizen-wavlm-large-s80-md"


def test_embed_moss_labels_weights_embeddings_by_turn_duration(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    embeddings = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(64000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(runner, "_embed_region", lambda *_: next(embeddings))

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [Turn(0.0, 1.0, "speaker_0"), Turn(1.0, 4.0, "speaker_0")],
    )

    np.testing.assert_allclose(
        pooled["speaker_0"], np.array([0.25, 0.75], dtype=np.float32)
    )
    assert pooled["speaker_0"].dtype == np.float32


def test_embed_moss_labels_keeps_weights_aligned_when_skipping_invalid(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    embeddings = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.array([np.nan, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(160000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(runner, "_embed_region", lambda *_: next(embeddings))

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [
            Turn(0.0, 1.0, "speaker_0"),
            Turn(1.0, 3.0, "speaker_0"),
            Turn(3.0, 7.0, "speaker_0"),
            Turn(7.0, 10.0, "speaker_0"),
        ],
    )

    np.testing.assert_allclose(
        pooled["speaker_0"], np.array([0.25, 0.75], dtype=np.float32)
    )


def test_embed_moss_labels_keeps_namespaced_labels_independent(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    embeddings = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(32000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(runner, "_embed_region", lambda *_: next(embeddings))

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [Turn(0.0, 1.0, "c000:S01"), Turn(1.0, 2.0, "c001:S01")],
    )

    assert set(pooled) == {"c000:S01", "c001:S01"}
    np.testing.assert_array_equal(
        pooled["c000:S01"], np.array([1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        pooled["c001:S01"], np.array([0.0, 1.0], dtype=np.float32)
    )


def test_embed_moss_labels_weights_boundary_overrun_by_clipped_duration(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    embeddings = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(runner, "_embed_region", lambda *_: next(embeddings))

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [Turn(0.75, 2.0, "speaker_0"), Turn(0.0, 0.75, "speaker_0")],
    )

    np.testing.assert_allclose(
        pooled["speaker_0"], np.array([0.25, 0.75], dtype=np.float32)
    )


def test_embed_moss_labels_floors_tiny_clipped_duration_weight(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    embeddings = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(runner, "_embed_region", lambda *_: next(embeddings))

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [Turn(0.9995, 2.0, "speaker_0"), Turn(0.0, 0.003, "speaker_0")],
    )

    np.testing.assert_allclose(
        pooled["speaker_0"], np.array([0.25, 0.75], dtype=np.float32)
    )


def test_embed_moss_labels_empty_input_does_not_load_audio(monkeypatch):
    import fusion_diarize.diarizen_runner as mod

    runner = object.__new__(mod.DiariZenRunner)

    def fail_if_called(_):
        raise AssertionError("load_mono16k must not be called")

    monkeypatch.setattr(mod, "load_mono16k", fail_if_called)

    assert runner.embed_moss_labels("audio.wav", []) == {}


def test_embed_moss_labels_excludes_overlapping_regions(monkeypatch):
    import fusion_diarize.diarizen_runner as mod
    from fusion_diarize.types import Turn

    runner = object.__new__(mod.DiariZenRunner)
    regions: list[tuple[float, float]] = []

    monkeypatch.setattr(
        mod,
        "load_mono16k",
        lambda _: (np.zeros(160000, dtype=np.float32), 16000),
    )

    def capture_region(_wav, start, end):
        regions.append((start, end))
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(runner, "_embed_region", capture_region)

    pooled = runner.embed_moss_labels(
        "audio.wav",
        [
            Turn(0.0, 4.0, "speaker_0"),
            Turn(2.0, 3.0, "speaker_1"),
        ],
    )

    # Overlap [2,3] is impure for both speakers; speaker_1 has no clean residue.
    assert "speaker_0" in pooled
    assert "speaker_1" not in pooled
    assert regions == [(0.0, 2.0), (3.0, 4.0)]
