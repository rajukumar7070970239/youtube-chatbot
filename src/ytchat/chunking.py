"""Step 2: split the long transcript into overlapping, retrievable chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(
    text: str,
    video_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split ``text`` into Documents with ``video_id`` and ``chunk`` metadata."""
    if not text.strip():
        raise ValueError("Transcript is empty; nothing to split.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text], metadatas=[{"video_id": video_id}])
    for i, doc in enumerate(docs):
        doc.metadata["chunk"] = i
    return docs
