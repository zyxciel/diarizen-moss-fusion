"""CLI entrypoints: ``fusion-diarize run`` and ``fusion-diarize eval``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from fusion_diarize.diarizen_runner import DiariZenRunner
    from fusion_diarize.moss_runner import MossRunner
    from fusion_diarize.pipeline import run_pipeline

    work_dir = Path(args.work_dir)
    diarizen = DiariZenRunner(repo_id=args.diarizen_repo)
    moss = MossRunner(model_path=args.moss_model)
    outs = run_pipeline(
        audio=Path(args.audio),
        work_dir=work_dir,
        mode=args.mode,
        diarizen_runner=diarizen,
        moss_runner=moss,
        tau=args.tau,
    )
    for name, path in sorted(outs.items()):
        print(f"{name}: {path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import json

    from fusion_diarize.eval_der import compute_der

    result = compute_der(args.hyp, args.ref, collar=args.collar)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fusion-diarize")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run DiariZen+MOSS fusion pipeline")
    run.add_argument("--audio", required=True, help="Input audio path")
    run.add_argument("--work-dir", required=True, help="Cache / output directory")
    run.add_argument(
        "--mode",
        choices=["a", "b", "both"],
        default="both",
        help="Fuse mode (default: both)",
    )
    run.add_argument("--moss-model", required=True, help="MOSS model path")
    run.add_argument(
        "--diarizen-repo",
        default="BUT-FIT/diarizen-wavlm-large-s80-md",
        help="DiariZen HuggingFace repo id",
    )
    run.add_argument("--tau", type=float, default=0.6, help="Mode A confidence threshold")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="Compute DER vs reference RTTM")
    ev.add_argument("--hyp", required=True, help="Hypothesis RTTM")
    ev.add_argument("--ref", required=True, help="Reference RTTM")
    ev.add_argument("--collar", type=float, default=0.25, help="Collar seconds")
    ev.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main(sys.argv[1:])
