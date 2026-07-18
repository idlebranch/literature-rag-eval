"""Single CLI entrypoint for V3 evaluation operations.

Usage:
    python -m src.eval.cli migrate --eval-csv outputs/answer_eval_manual.csv \
        --judge-csv outputs/answer_judge_summary.csv \
        --run-id 2026-05-24_v1_gpt-oss-120b_claude-judge \
        --rag-model gpt-oss-120b --rag-prompt v1 --judge-model claude-opus-4-7

    python -m src.eval.cli export --run outputs/runs/<id>.json
    python -m src.eval.cli badcase --run outputs/runs/<id>.json
    python -m src.eval.cli index
    python -m src.eval.cli health
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.eval.badcase import write_badcase_report
from src.eval.export import to_csv, to_index_md, to_markdown
from src.eval.io import iter_runs, load_run, migrate_from_csv, save_run
from src.eval.schema import RunConfig

RUNS_DIR = Path("outputs/runs")
VIEWS_DIR = Path("outputs/views")
INDEX_PATH = Path("outputs/INDEX.md")


def cmd_migrate(args: argparse.Namespace) -> int:
    config = RunConfig(
        rag_model=args.rag_model or "",
        rag_prompt_version=args.rag_prompt or "",
        embedding_model=args.embedding_model or "",
        top_k=args.top_k,
        judge_model=args.judge_model or "",
        groundtruth_file=args.groundtruth or "",
    )
    run = migrate_from_csv(
        eval_csv=Path(args.eval_csv),
        judge_csv=Path(args.judge_csv) if args.judge_csv else None,
        run_id=args.run_id,
        config=config,
        judge_model=args.judge_model or "",
    )
    out_path = RUNS_DIR / f"{args.run_id}.json"
    save_run(run, out_path)
    s = run.summary
    print(f"[migrate] wrote {out_path}")
    print(
        f"  n_questions={s.n_questions} n_judged={s.n_judged} n_errored={s.n_errored} "
        f"avg_overall={s.avg_overall} badcases={s.badcase_count}"
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    run = load_run(Path(args.run))
    base = args.out_prefix or run.run_id
    fmt = args.format
    written: List[Path] = []
    if fmt in ("md", "both"):
        p = to_markdown(run, VIEWS_DIR / f"{base}.md", badcase_threshold=args.badcase_threshold)
        written.append(p)
    if fmt in ("csv", "both"):
        p = to_csv(run, VIEWS_DIR / f"{base}.csv")
        written.append(p)
    for p in written:
        print(f"[export] wrote {p}")
    return 0


def cmd_badcase(args: argparse.Namespace) -> int:
    run = load_run(Path(args.run))
    base = args.out_prefix or run.run_id
    p = write_badcase_report(
        run, VIEWS_DIR / f"{base}_badcase.md", badcase_threshold=args.badcase_threshold
    )
    print(f"[badcase] wrote {p}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    runs = [r for _, r in iter_runs(Path(args.runs_dir))]
    p = to_index_md(runs, Path(args.output))
    print(f"[index] wrote {p} ({len(runs)} runs)")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from src.eval.health import check_llm

    status = check_llm()
    print(status)
    return 0 if status.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from src.eval.runner import run_rag

    out_path = RUNS_DIR / f"{args.run_id}.json"
    run = run_rag(
        groundtruth=Path(args.groundtruth),
        run_id=args.run_id,
        output_path=out_path,
        prompt_version=args.prompt_version,
        sleep_secs=args.sleep,
        limit=args.limit,
        skip_preflight=args.skip_preflight,
    )
    s = run.summary
    print(f"[run] wrote {out_path}")
    print(
        f"  n_questions={s.n_questions} n_errored={s.n_errored} "
        f"(prompt_hash={run.config.rag_prompt_hash})"
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from src.eval.runner import judge_run

    run = judge_run(
        run_path=Path(args.run),
        judge_model=args.judge_model or None,
        sleep_secs=args.sleep,
        only_missing=not args.rejudge_all,
    )
    s = run.summary
    print(f"[judge] updated {args.run}")
    print(
        f"  n_judged={s.n_judged}/{s.n_questions} n_errored={s.n_errored} "
        f"avg_overall={s.avg_overall} badcases={s.badcase_count}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.eval.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("migrate", help="Build a JSON run from legacy eval+judge CSVs")
    pm.add_argument("--eval-csv", required=True)
    pm.add_argument("--judge-csv", default=None)
    pm.add_argument("--run-id", required=True)
    pm.add_argument("--rag-model", default="")
    pm.add_argument("--rag-prompt", default="")
    pm.add_argument("--embedding-model", default="")
    pm.add_argument("--top-k", type=int, default=5)
    pm.add_argument("--judge-model", default="")
    pm.add_argument("--groundtruth", default="")
    pm.set_defaults(func=cmd_migrate)

    pe = sub.add_parser("export", help="Generate readable MD / flat CSV from a run JSON")
    pe.add_argument("--run", required=True)
    pe.add_argument("--format", choices=["md", "csv", "both"], default="both")
    pe.add_argument("--out-prefix", default=None)
    pe.add_argument("--badcase-threshold", type=int, default=4)
    pe.set_defaults(func=cmd_export)

    pb = sub.add_parser("badcase", help="Generate badcase analysis from a run JSON")
    pb.add_argument("--run", required=True)
    pb.add_argument("--out-prefix", default=None)
    pb.add_argument("--badcase-threshold", type=int, default=4)
    pb.set_defaults(func=cmd_badcase)

    pi = sub.add_parser("index", help="Refresh outputs/INDEX.md across all runs")
    pi.add_argument("--runs-dir", default=str(RUNS_DIR))
    pi.add_argument("--output", default=str(INDEX_PATH))
    pi.set_defaults(func=cmd_index)

    ph = sub.add_parser("health", help="LLM preflight check (api key + model)")
    ph.set_defaults(func=cmd_health)

    pr = sub.add_parser("run", help="Run RAG against groundtruth → JSON")
    pr.add_argument("--groundtruth", required=True)
    pr.add_argument("--run-id", required=True)
    pr.add_argument("--prompt-version", default="")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--sleep", type=float, default=0.5)
    pr.add_argument("--skip-preflight", action="store_true",
                    help="Skip the LLM health ping. Useful if your endpoint blocks pings.")
    pr.set_defaults(func=cmd_run)

    pj = sub.add_parser("judge", help="Add LLM-as-judge scores to an existing run JSON")
    pj.add_argument("--run", required=True)
    pj.add_argument("--judge-model", default="")
    pj.add_argument("--sleep", type=float, default=0.5)
    pj.add_argument("--rejudge-all", action="store_true",
                    help="Re-judge every question (default: only those missing a judge).")
    pj.set_defaults(func=cmd_judge)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
