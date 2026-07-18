from pathlib import Path
from tqdm import tqdm
from src.config import settings
from src.pdf_loader import load_pdfs
from src.chunking import build_chunks
from src.embedder import embed_texts
from src.vectorstore import add_chunks
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading PDFs from: %s", settings.pdf_dir)
    pages = load_pdfs(settings.pdf_dir)
    logger.info("Loaded pages: %d", len(pages))

    if not pages:
        raise RuntimeError("No extractable PDF text found. Check data/pdfs or avoid scanned PDFs.")

    chunks = build_chunks(pages)
    logger.info("Built chunks: %d", len(chunks))

    texts = [c["text"] for c in chunks]

    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        embeddings.extend(embed_texts(texts[i : i + batch_size]))

    add_chunks(chunks, embeddings, reset=True)

    logger.info("Done. Chroma collection rebuilt.")
    logger.info("Collection name: %s", settings.collection_name)
    logger.info("Chroma dir: %s", settings.chroma_dir)


if __name__ == "__main__":
    main()
