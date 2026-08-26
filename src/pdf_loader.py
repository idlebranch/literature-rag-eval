from pathlib import Path
from typing import Dict, List
import fitz
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_pdfs(pdf_dir: str) -> List[Dict]:
    """Load text from all PDFs in a directory using PyMuPDF.

    Returns page-level records:
    {
      "paper_id": filename without suffix,
      "source": filename,
      "page": page number starting from 1,
      "text": extracted text
    }
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    records: List[Dict] = []

    pdf_files = sorted(pdf_path.rglob("*.pdf"))
    logger.info("Found PDF files: %d", len(pdf_files))

    for file in pdf_files:
        paper_id = file.stem
        logger.info("Parsing: %s", file.name)

        try:
            doc = fitz.open(file)
        except Exception as e:
            logger.warning("Cannot open %s: %s", file.name, e)
            continue

        for page_index in range(len(doc)):
            try:
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
                text = " ".join(text.split())

                if len(text) < 50:
                    continue

                records.append(
                    {
                        "paper_id": paper_id,
                        "source": file.name,
                        "page": page_index + 1,
                        "text": text,
                    }
                )
            except Exception as e:
                logger.warning("%s page %d: %s", file.name, page_index + 1, e)
                continue

        doc.close()

    return records


def load_pdf_pages(pdf_path: str) -> List[Dict]:
    """Extract line-preserving per-page text from a single PDF.

    Unlike :func:`load_pdfs` (which collapses whitespace per page for the legacy
    fixed chunker), this keeps newlines so section-aware chunking can detect
    heading lines and track page provenance.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    records: List[Dict] = []
    try:
        for page_index in range(len(doc)):
            try:
                text = doc.load_page(page_index).get_text("text") or ""
                records.append(
                    {
                        "paper_id": pdf_path.stem,
                        "source": pdf_path.name,
                        "page": page_index + 1,
                        "text": text,
                    }
                )
            except Exception as e:
                logger.warning("%s page %d: %s", pdf_path.name, page_index + 1, e)
    finally:
        doc.close()
    return records
