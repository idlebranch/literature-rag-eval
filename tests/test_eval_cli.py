"""Tests for src.eval.cli argument wiring (no heavy ops)."""
from src.eval.cli import build_parser


def test_parser_has_all_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "cmd"]
    assert actions
    choices = set(actions[0].choices.keys())
    assert {"migrate", "export", "badcase", "index", "health", "run", "judge"} <= choices


def test_migrate_required_args():
    parser = build_parser()
    ns = parser.parse_args([
        "migrate",
        "--eval-csv", "a.csv",
        "--run-id", "rid",
    ])
    assert ns.cmd == "migrate"
    assert ns.eval_csv == "a.csv"
    assert ns.run_id == "rid"
    assert ns.judge_csv is None
    assert ns.top_k == 5


def test_export_defaults_both_format():
    parser = build_parser()
    ns = parser.parse_args(["export", "--run", "x.json"])
    assert ns.format == "both"
    assert ns.badcase_threshold == 4
