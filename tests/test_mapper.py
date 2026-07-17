import numpy as np
from fusion_diarize.types import Turn
from fusion_diarize.mapper import map_moss_speakers, assignment_confidence, remap_turns


def test_map_by_overlap_and_embedding():
    diarizen = [
        Turn(0.0, 5.0, "speaker_0"),
        Turn(5.0, 10.0, "speaker_1"),
    ]
    moss = [
        Turn(0.2, 4.8, "S01", text="a"),
        Turn(5.2, 9.5, "S02", text="b"),
    ]
    centroids = {
        "speaker_0": np.array([1.0, 0.0]),
        "speaker_1": np.array([0.0, 1.0]),
    }
    moss_emb = {
        "S01": np.array([0.9, 0.1]),
        "S02": np.array([0.1, 0.9]),
    }
    mapping = map_moss_speakers(moss, diarizen, moss_emb, centroids)
    assert mapping["S01"] == "speaker_0"
    assert mapping["S02"] == "speaker_1"


def test_low_confidence_when_ambiguous():
    conf = assignment_confidence(overlap=0.1, cosine=0.2)
    assert conf < 0.5


def test_remap_turns():
    moss = [
        Turn(0.2, 4.8, "S01", text="a"),
        Turn(5.2, 9.5, "S02", text="b"),
    ]
    mapping = {"S01": "speaker_0", "S02": "speaker_1"}
    remapped = remap_turns(moss, mapping)
    assert remapped[0].speaker_id == "speaker_0"
    assert remapped[1].speaker_id == "speaker_1"
    assert remapped[0].text == "a"
    assert remapped[1].text == "b"
