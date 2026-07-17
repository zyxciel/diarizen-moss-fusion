"""Audio load/save helpers at 16 kHz mono.

Prefers torchaudio when installed; falls back to stdlib ``wave`` (PCM16)
plus linear resample so unit tests work without GPU deps.
"""
from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

_SAMPLE_RATE = 16000


def write_mono16k_wav(path: Path, waveform: np.ndarray) -> None:
    assert waveform.ndim == 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torchaudio

        tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
        torchaudio.save(str(path), tensor, _SAMPLE_RATE)
        return
    except ImportError:
        pass

    clipped = np.clip(waveform.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def load_mono16k(path: Path) -> tuple[np.ndarray, int]:
    try:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != _SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
            sr = _SAMPLE_RATE
        return wav.squeeze(0).numpy(), int(sr)
    except ImportError:
        pass

    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    if nchannels > 1:
        pcm = pcm.reshape(-1, nchannels).mean(axis=1)

    if sr != _SAMPLE_RATE:
        # Linear resample for stdlib fallback (tests use 16 kHz files).
        duration = len(pcm) / float(sr)
        n_out = int(round(duration * _SAMPLE_RATE))
        if len(pcm) == 0 or n_out == 0:
            pcm = np.zeros(0, dtype=np.float32)
        else:
            x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
        sr = _SAMPLE_RATE

    return pcm.astype(np.float32), int(sr)


def probe_duration(path: Path) -> float:
    wav, sr = load_mono16k(path)
    return float(len(wav) / sr)


def slice_wav(path: Path, start: float, end: float, out_path: Path) -> Path:
    wav, sr = load_mono16k(path)
    i0 = max(0, int(round(start * sr)))
    i1 = min(len(wav), int(round(end * sr)))
    write_mono16k_wav(out_path, wav[i0:i1])
    return out_path
