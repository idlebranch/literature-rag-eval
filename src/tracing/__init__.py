"""Request-level trace spine (milestone 1: POST /chat only).

Public entrypoint is :func:`src.tracing.instrumentation.traced_chat`, which wraps
a single RAG call and emits exactly one trace record to
``<project_root>/outputs/traces/traces.jsonl``.

Kept deliberately dependency-free (stdlib only) and independent of ``src.eval``.
"""
