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
