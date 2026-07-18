import subprocess
import sys
from pathlib import Path

import streamlit as st
from src.config import settings
from src.rag_chain import answer_question
from src.vectorstore import get_collection_count

st.set_page_config(page_title="Literature RAG Evaluation MVP", layout="wide")

st.title("Scientific Literature RAG Evaluation Platform")
st.caption("代码版替代 Dify：PDF → Chroma → Retrieval → LLM Answer → Citation")

# ── Sidebar: Index Status ──────────────────────────────────────────────
st.sidebar.header("📊 Index Status")

# PDF count
pdf_dir = Path(settings.pdf_dir)
pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.exists() else 0
st.sidebar.metric("📄 PDF Files", pdf_count)

# ChromaDB status
chroma_db_path = Path(settings.chroma_dir)
chroma_sqlite = chroma_db_path / "chroma.sqlite3"
chroma_exists = chroma_sqlite.exists() and chroma_sqlite.stat().st_size > 0

if chroma_exists:
    chunk_count = get_collection_count()
    st.sidebar.success(f"✅ ChromaDB: {chunk_count:,} chunks")
else:
    st.sidebar.warning("⚠️ ChromaDB not built")

# Top-K display
st.sidebar.metric("🔍 Default Top-K", settings.top_k)

# Rebuild hint
if pdf_count > 0:
    st.sidebar.info("💡 If you add, remove, or replace PDFs, rebuild the index below.")
elif not chroma_exists:
    st.sidebar.info("💡 Put PDFs in data/pdfs/, then rebuild the index.")

# Rebuild index button
if st.sidebar.button("🔄 Rebuild Index", use_container_width=True, help="Re-ingest all PDFs into ChromaDB"):
    if pdf_count == 0:
        st.sidebar.error("No PDFs found in data/pdfs/")
    else:
        with st.sidebar:
            with st.spinner(f"Rebuilding index from {pdf_count} PDFs... This may take several minutes."):
                result = subprocess.run(
                    [sys.executable, "-m", "src.ingest"],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                new_count = get_collection_count()
                st.success(f"✅ Rebuild complete: {new_count:,} chunks indexed")
            else:
                st.error(f"❌ Rebuild failed (exit {result.returncode})")
                if result.stderr:
                    st.text(result.stderr[-1000:])

st.sidebar.divider()

# ── Main: Q&A ──────────────────────────────────────────────────────────
question = st.text_area("输入你的问题", value="PAC 去除抗生素的主要机制是什么？", height=100)
top_k = st.slider("Top-K", min_value=3, max_value=15, value=settings.top_k)

if st.button("Run RAG", type="primary"):
    with st.spinner("Retrieving and generating..."):
        result = answer_question(question, top_k=top_k)

    st.subheader("📝 Answer")
    st.write(result["answer"])

    st.subheader("📚 Retrieved Contexts")
    st.caption(f"Showing {len(result['contexts'])} sources ordered by relevance")

    for i, hit in enumerate(result["contexts"], start=1):
        meta = hit["metadata"]
        distance = hit["distance"]

        # Relevance level based on cosine distance (BGE-M3: lower = more similar)
        if distance < 0.3:
            relevance = "🟢 High"
        elif distance < 0.7:
            relevance = "🟡 Medium"
        else:
            relevance = "🔴 Low"

        # Clean filename without path
        filename = meta.get("source", "unknown")
        page = meta.get("page", "?")

        with st.expander(f"[S{i}] {filename}  |  Page {page}  |  Relevance: {relevance}"):
            st.caption(f"Chunk index: {meta.get('chunk_index', '?')}  ·  Distance: {distance:.4f}")
            st.markdown(f"> {hit['text']}")
