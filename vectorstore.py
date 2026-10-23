"""Step 3: embed chunks and keep them in a (locally cached) FAISS vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def get_embeddings(model: str, api_key: str | None = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, api_key=api_key or None)


def build_vector_store(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    video_id: str,
    cache_dir: str = ".cache/vectorstores",
) -> FAISS:
    """Embed ``docs`` and persist the index so the same video is only embedded once."""
    store = FAISS.from_documents(docs, embeddings)
    path = Path(cache_dir) / video_id
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store


def load_vector_store(
    embeddings: OpenAIEmbeddings, video_id: str, cache_dir: str = ".cache/vectorstores"
) -> FAISS | None:
    path = Path(cache_dir) / video_id
    if not (path / "index.faiss").exists():
        return None
    # The index was written by this app, so deserialising it is safe.
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
