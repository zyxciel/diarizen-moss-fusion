"""Tests for fusion_diarize.eval_der.compute_der."""
from __future__ import annotations

from fusion_diarize.eval_der import compute_der
from fusion_diarize.export import write_rttm
from fusion_diarize.types import Turn


def test_identical_rttm_der_near_zero(tmp_path):
    turns = [
        Turn(0.0, 2.0, "speaker_0"),
        Turn(2.5, 4.0, "speaker_1"),
    ]
    hyp = tmp_path / "hyp.rttm"
    ref = tmp_path / "ref.rttm"
    write_rttm(turns, hyp, uri="utt")
    write_rttm(turns, ref, uri="utt")

    out = compute_der(hyp, ref, collar=0.25)
    assert out["der"] == 0.0 or abs(out["der"]) < 1e-9
    assert out["collar"] == 0.25
    assert "false_alarm" in out
    assert "missed_detection" in out
    assert "confusion" in out


def test_clear_miss_der_positive(tmp_path):
    ref_turns = [Turn(0.0, 3.0, "speaker_0")]
    hyp_turns: list[Turn] = []
    hyp = tmp_path / "hyp.rttm"
    ref = tmp_path / "ref.rttm"
    write_rttm(hyp_turns, hyp, uri="utt")
    write_rttm(ref_turns, ref, uri="utt")

    out = compute_der(hyp, ref, collar=0.25)
    assert out["der"] > 0.0
    assert out["missed_detection"] > 0.0
