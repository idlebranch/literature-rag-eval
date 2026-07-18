"""V3 evaluation subpackage.

JSON-first data model + CLI + Streamlit dashboard.

Layout:
    schema.py    Dataclasses for an EvalRun (canonical truth).
    io.py        JSON load/save + migration from legacy CSVs.
    export.py    Generate readable MD + flat CSV views from JSON.
    badcase.py   Group badcases by error_type and propose fixes.
    health.py    API key / endpoint preflight check.
    cli.py       Single CLI entrypoint: python -m src.eval.cli <subcommand>.
"""
