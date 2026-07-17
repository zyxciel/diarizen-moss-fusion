# tests/test_export.py
from fusion_diarize.types import Turn, DiarResult, AsrStatus, Source
from fusion_diarize.export import write_rttm, write_json, read_json, read_rttm_as_turns

def test_json_roundtrip(tmp_path):
    turns = [
        Turn(0.0, 1.0, "speaker_0", text="hi", asr_status=AsrStatus.PROVISIONAL,
             source=Source.FUSED, confidence=0.9),
        Turn(1.5, 2.0, "speaker_1", text="", asr_status=AsrStatus.EMPTY,
             source=Source.DIARIZEN, confidence=0.5),
    ]
    result = DiarResult(
        turns=turns,
        meta={"mode": "a"},
        centroids={"speaker_0": [1.0, 0.0]},
    )
    path = tmp_path / "out.json"
    write_json(result, path)
    loaded = read_json(path)
    assert len(loaded.turns) == 2
    assert loaded.turns[0].speaker_id == "speaker_0"
    assert loaded.turns[0].text == "hi"
    assert loaded.turns[0].asr_status == AsrStatus.PROVISIONAL
    assert loaded.turns[0].source == Source.FUSED
    assert loaded.turns[0].confidence == 0.9
    assert loaded.turns[1].asr_status == AsrStatus.EMPTY
    assert loaded.turns[1].source == Source.DIARIZEN
    assert loaded.meta["mode"] == "a"
    assert loaded.centroids == {"speaker_0": [1.0, 0.0]}

def test_rttm_roundtrip(tmp_path):
    turns = [Turn(1.0, 3.5, "speaker_0")]
    path = tmp_path / "out.rttm"
    write_rttm(turns, path, uri="utt1")
    loaded = read_rttm_as_turns(path)
    assert len(loaded) == 1
    assert abs(loaded[0].start - 1.0) < 1e-6
    assert abs(loaded[0].end - 3.5) < 1e-6
    assert loaded[0].speaker_id == "speaker_0"
