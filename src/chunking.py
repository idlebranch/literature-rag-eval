from typing import Dict, List
from src.config import settings


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking.

    First version deliberately keeps it simple.
    Later you can replace this with section-aware chunking.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def build_chunks(page_records: List[Dict]) -> List[Dict]:
    all_chunks: List[Dict] = []

    for rec in page_records:
        chunks = chunk_text(
            rec["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for idx, chunk in enumerate(chunks):
            chunk_id = f'{rec["paper_id"]}_p{rec["page"]}_c{idx}'
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "paper_id": rec["paper_id"],
                        "source": rec["source"],
                        "page": rec["page"],
                        "chunk_index": idx,
                    },
                }
            )

    return all_chunks
