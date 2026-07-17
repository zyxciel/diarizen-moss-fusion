"""Offline pipeline test with FakeDiariZen + FakeMoss (no GPU / HF)."""
from pathlib import Path

import numpy as np

from fusion_diarize.audio_prep import write_mono16k_wav
from fusion_diarize.diarizen_runner import FakeDiariZenRunner
from fusion_diarize.export import read_json
from fusion_diarize.moss_runner import FakeMossRunner
from fusion_diarize.pipeline import run_pipeline
from fusion_diarize.types import AsrStatus, Source, Turn


def test_run_pipeline_mode_both_writes_exports(tmp_path: Path):
    audio = tmp_path / "utt.wav"
    write_mono16k_wav(audio, np.zeros(16000, dtype=np.float32))  # 1 s

    diarizen_turns = [
        Turn(0.0, 0.6, "speaker_0"),
        Turn(0.4, 1.0, "speaker_1"),
    ]
    centroids = {
        "speaker_0": np.array([1.0, 0.0], dtype=np.float32),
        "speaker_1": np.array([0.0, 1.0], dtype=np.float32),
    }
    moss_turns = [
        Turn(
            0.05,
            0.55,
            "S01",
            text="hello",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
        Turn(
            0.45,
            0.95,
            "S02",
            text="world",
            asr_status=AsrStatus.PROVISIONAL,
            source=Source.MOSS,
        ),
    ]

    work_dir = tmp_path / "work"
    outs = run_pipeline(
        audio=audio,
        work_dir=work_dir,
        mode="both",
        diarizen_runner=FakeDiariZenRunner(diarizen_turns, centroids),
        moss_runner=FakeMossRunner(turns_by_default=moss_turns),
        tau=0.6,
    )

    assert outs["mode_a.json"].is_file()
    assert outs["mode_b.json"].is_file()
    assert outs["mode_a.rttm"].is_file()
    assert outs["mode_b.rttm"].is_file()

    a = read_json(outs["mode_a.json"])
    b = read_json(outs["mode_b.json"])
    assert a.meta.get("mode") == "a"
    assert b.meta.get("mode") == "b"
    assert isinstance(a.turns, list)
    assert isinstance(b.turns, list)

    # caches written
    assert (work_dir / "diarizen.json").is_file()
    assert (work_dir / "chunks.json").is_file()
    assert (work_dir / "moss_turns.json").is_file()
    assert (work_dir / "prepared.wav").is_file()
