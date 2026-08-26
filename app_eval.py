import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="RAG Eval Dashboard", page_icon="📊", layout="wide")
st.title("RAG Evaluation Dashboard")


def _find_eval_root() -> Path:
    current = Path("outputs")
    if list((current / "views").glob("*.csv")):
        return current
    archives = sorted(
        (path for path in current.glob("archive_*") if list((path / "views").glob("*.csv"))),
        reverse=True,
    )
    return archives[0] if archives else current


EVAL_ROOT = _find_eval_root()
VIEWS = EVAL_ROOT / "views"
INDEX = EVAL_ROOT / "INDEX.md"
st.caption(f"Data source: `{EVAL_ROOT}`")

# ── Column name mapping (view CSV → dashboard canonical names) ──
_COL_MAP = {
    "qid": "id",
    "faithfulness": "faithfulness_score",
    "completeness": "completeness_score",
    "citation": "citation_score",
    "overall": "overall_score",
}


def _load_view_csv(path: Path) -> pd.DataFrame:
    """Load a view CSV and normalise column names to the canonical form."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
    return df


# ── Auto-discover available CSVs in outputs/views/ ──
if VIEWS.exists():
    _csv_paths = sorted(VIEWS.glob("*.csv"))
else:
    _csv_paths = []

if not _csv_paths:
    st.error("No evaluation CSV files found in `outputs/views/`.")
    st.info(
        "Run the evaluation pipeline first to populate `outputs/views/` "
        "with per‑run summary CSV files.\n\n"
        "Expected naming pattern: `<date>_<run_id>.csv`"
    )
    st.stop()

# Load runs (one DataFrame per discovered CSV)
_run_map: dict[str, pd.DataFrame] = {}
for _p in _csv_paths:
    _run_map[_p.stem] = _load_view_csv(_p)

_run_names = sorted(_run_map.keys(), reverse=True)  # newest first

# ── Sidebar: run selection ──
st.sidebar.header("Run Selection")
_sel_v1 = st.sidebar.selectbox("Primary run (V1)", _run_names, index=0)
_sel_v2 = st.sidebar.selectbox(
    "Secondary run (V2) — optional (comparison / badcase)",
    ["(none)"] + _run_names,
    index=1 if len(_run_names) > 1 else 0,
)

v1 = _run_map[_sel_v1]
v2 = _run_map[_sel_v2] if _sel_v2 != "(none)" and _sel_v2 in _run_map else None

# ── Auto-discover MD reports in outputs/views/ ──
_MD_MAP: dict[str, Path] = {}
if VIEWS.exists():
    for _md in sorted(VIEWS.glob("*.md")):
        _MD_MAP[_md.stem] = _md


def _md_section(header: str, stem: str):
    """Render a markdown section if the file exists in views/."""
    st.header(header)
    _path = _MD_MAP.get(stem)
    if _path and _path.exists():
        st.markdown(_path.read_text(encoding="utf-8", errors="ignore"))
    else:
        st.info(f"Markdown report `{stem}.md` not found in `outputs/views/`.")


# ═══════════════════════════════════════════════════════════════
# 1. Retrieval Evaluation
# ═══════════════════════════════════════════════════════════════
st.header("1. Retrieval Evaluation")

if v1 is not None:
    if "hit_at_k" in v1.columns:
        hit = v1["hit_at_k"].mean()
        st.metric("Retrieval Hit@K", f"{hit:.3f}")
    elif "n_retrieved" in v1.columns:
        # view CSV provides per‑question n_retrieved; approximate retrieval health
        mean_n = v1["n_retrieved"].mean()
        st.metric("Avg chunks retrieved", f"{mean_n:.1f}")
    st.dataframe(v1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 2. Answer Judge Summary
# ═══════════════════════════════════════════════════════════════
st.header("2. Answer Judge Summary")

metrics = ["faithfulness_score", "completeness_score", "citation_score", "overall_score"]


def _show_score_panel(df: pd.DataFrame, label: str):
    st.subheader(label)
    cols = st.columns(4)
    for i, m in enumerate(metrics):
        if m in df.columns:
            cols[i].metric(m, f"{df[m].mean():.2f}")


if v1 is not None:
    _show_score_panel(v1, f"{_sel_v1} Mean Scores")

if v2 is not None:
    _show_score_panel(v2, f"{_sel_v2} Mean Scores")

# ═══════════════════════════════════════════════════════════════
# 3. V1 vs V2 Comparison
# ═══════════════════════════════════════════════════════════════
if v1 is not None and v2 is not None:
    st.header("3. V1 vs V2 Comparison")

    _id_col = "id"  # normalised during load
    if _id_col in v1.columns and _id_col in v2.columns:
        comp = v1[[_id_col, "question", "overall_score"]].merge(
            v2[[_id_col, "overall_score"]],
            on=_id_col,
            suffixes=("_v1", "_v2"),
        )
        comp["delta"] = comp["overall_score_v2"] - comp["overall_score_v1"]
        st.dataframe(comp, use_container_width=True)

    mean_compare = pd.DataFrame({
        "metric": metrics,
        "v1_mean": [v1[m].mean() if m in v1.columns else None for m in metrics],
        "v2_mean": [v2[m].mean() if m in v2.columns else None for m in metrics],
    })
    mean_compare["delta"] = mean_compare["v2_mean"] - mean_compare["v1_mean"]
    st.dataframe(mean_compare, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 4. Badcase Explorer
# ═══════════════════════════════════════════════════════════════
st.header("4. Badcase Explorer")

target = v2 if v2 is not None else v1

if target is not None:
    if "error_type" in target.columns:
        error_types = ["ALL"] + sorted(
            target["error_type"].dropna().astype(str).unique().tolist()
        )
    else:
        error_types = ["ALL"]
    selected_error = st.selectbox("Filter by error_type", error_types)
    max_score = st.slider("Max overall_score", 1, 5, 5)
    keyword = st.text_input("Keyword in question")

    filtered = target.copy()
    if selected_error != "ALL" and "error_type" in filtered.columns:
        filtered = filtered[filtered["error_type"].astype(str) == selected_error]
    if "overall_score" in filtered.columns:
        filtered = filtered[filtered["overall_score"] <= max_score]
    if keyword:
        filtered = filtered[
            filtered["question"].astype(str).str.contains(keyword, case=False, na=False)
        ]

    st.dataframe(filtered, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 5. Badcase Report
# ═══════════════════════════════════════════════════════════════
# Choose the badcase MD that matches the selected secondary (or primary) run
_badcase_stem = None
if _sel_v2 != "(none)" and _sel_v2 in _run_map:
    _badcase_stem = f"{_sel_v2}_badcase"
elif _sel_v1 in _run_map:
    _badcase_stem = f"{_sel_v1}_badcase"

if _badcase_stem and _badcase_stem in _MD_MAP:
    _md_section("5. Badcase Report", _badcase_stem)
else:
    st.header("5. Badcase Report")
    st.info("No matching badcase report found in `outputs/views/`.")

# ═══════════════════════════════════════════════════════════════
# 6. Index / Overview
# ═══════════════════════════════════════════════════════════════
st.header("6. Evaluation Index")
if INDEX.exists():
    st.markdown(INDEX.read_text(encoding="utf-8", errors="ignore"))
else:
    st.info("`outputs/INDEX.md` not found. Run the evaluation pipeline to generate it.")
