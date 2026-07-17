import numpy as np
from fusion_diarize.audio_prep import probe_duration, write_mono16k_wav, load_mono16k, slice_wav


def test_write_and_probe(tmp_path):
    path = tmp_path / "a.wav"
    write_mono16k_wav(path, np.zeros(24000, dtype=np.float32))
    assert abs(probe_duration(path) - 1.5) < 1e-3
    wav, sr = load_mono16k(path)
    assert sr == 16000
    assert wav.ndim == 1
    assert len(wav) == 24000


def test_slice_wav(tmp_path):
    path = tmp_path / "full.wav"
    # 1.0s ramp so slice content is identifiable
    full = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    write_mono16k_wav(path, full)
    out = tmp_path / "slice.wav"
    result = slice_wav(path, 0.25, 0.75, out)
    assert result == out
    wav, sr = load_mono16k(out)
    assert sr == 16000
    assert abs(len(wav) - 8000) <= 1
    assert abs(float(wav[0]) - float(full[4000])) < 1e-4
