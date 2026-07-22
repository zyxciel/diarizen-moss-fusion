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
    moss = MossRunner(
        model_path=args.moss_model,
        max_new_tokens=args.max_new_tokens,
    )
    outs = run_pipeline(
        audio=Path(args.audio),
        work_dir=work_dir,
        mode=args.mode,
        diarizen_runner=diarizen,
        moss_runner=moss,
        tau=args.tau,
        target_min=args.chunk_min,
        target_max=args.chunk_max,
        hard_cap=args.chunk_max,
        overlap=args.chunk_overlap,
        force_rechunk=args.force_rechunk,
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
    from fusion_diarize.chunk_planner import (
        DEFAULT_OVERLAP,
        DEFAULT_TARGET_MAX,
        DEFAULT_TARGET_MIN,
    )
    from fusion_diarize.moss_runner import DEFAULT_MAX_NEW_TOKENS

    p = argparse.ArgumentParser(prog="fusion-diarize")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run DiariZen+MOSS fusion pipeline")
    run.add_argument("--audio", required=True, help="Input audio path")
    run.add_argument("--work-dir", required=True, help="Cache / output directory")
    run.add_argument(
        "--mode",
        choices=["a", "b", "c", "both"],
        default="c",
        help="Fuse mode (default: c — MOSS-primary with DiariZen guardrails)",
    )
    run.add_argument("--moss-model", required=True, help="MOSS model path")
    run.add_argument(
        "--diarizen-repo",
        default="BUT-FIT/diarizen-wavlm-large-s80-md",
        help="DiariZen HuggingFace repo id",
    )
    run.add_argument("--tau", type=float, default=0.6, help="Mode A confidence threshold")
    run.add_argument(
        "--chunk-min",
        type=float,
        default=DEFAULT_TARGET_MIN,
        help=f"Min chunk length seconds before cut search (default {DEFAULT_TARGET_MIN})",
    )
    run.add_argument(
        "--chunk-max",
        type=float,
        default=DEFAULT_TARGET_MAX,
        help=f"Max MOSS chunk length seconds (default {DEFAULT_TARGET_MAX} = 20 min)",
    )
    run.add_argument(
        "--chunk-overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Overlap between adjacent chunks seconds (default {DEFAULT_OVERLAP})",
    )
    run.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"MOSS generation budget (default {DEFAULT_MAX_NEW_TOKENS})",
    )
    run.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Ignore cached chunks.json / moss_turns.json and replan",
    )
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
