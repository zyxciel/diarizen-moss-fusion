from fusion_diarize.cli import _cmd_run, build_parser
from fusion_diarize import diarizen_runner, moss_runner, pipeline


def _run_args(*extra: str):
    return build_parser().parse_args(
        [
            "run",
            "--audio",
            "input.wav",
            "--work-dir",
            "work",
            "--moss-model",
            "model",
            *extra,
        ]
    )


def test_identity_map_defaults_to_hierarchical():
    assert _run_args().identity_map == "hierarchical"


def test_identity_map_accepts_legacy_override():
    assert _run_args("--identity-map", "legacy").identity_map == "legacy"


def test_run_command_forwards_identity_map(monkeypatch):
    captured = {}

    class FakeDiariZen:
        def __init__(self, **kwargs):
            pass

    class FakeMoss:
        def __init__(self, **kwargs):
            pass

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(diarizen_runner, "DiariZenRunner", FakeDiariZen)
    monkeypatch.setattr(moss_runner, "MossRunner", FakeMoss)
    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    assert _cmd_run(_run_args("--identity-map", "legacy")) == 0
    assert captured["identity_map"] == "legacy"
