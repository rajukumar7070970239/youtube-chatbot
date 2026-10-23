import pytest

from ytchat.chunking import split_transcript
from ytchat.utils import extract_video_id


@pytest.mark.parametrize(
    "value",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=10s",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(value):
    assert extract_video_id(value) == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    with pytest.raises(ValueError):
        extract_video_id("not a video")


def test_split_transcript_chunks_and_metadata():
    text = "word " * 2000
    docs = split_transcript(text, "abc12345678", chunk_size=1000, chunk_overlap=200)
    assert len(docs) > 1
    assert all(len(d.page_content) <= 1000 for d in docs)
    assert docs[0].metadata == {"video_id": "abc12345678", "chunk": 0}


def test_split_transcript_empty():
    with pytest.raises(ValueError):
        split_transcript("   ", "abc12345678")
