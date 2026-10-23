"""Step 1: fetch a YouTube transcript and flatten it into one text string."""

from __future__ import annotations

from typing import Iterable

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

try:  # present in newer versions of youtube-transcript-api
    from youtube_transcript_api import VideoUnavailable
except ImportError:  # pragma: no cover
    VideoUnavailable = TranscriptsDisabled  # type: ignore


class TranscriptError(RuntimeError):
    """Raised when a transcript cannot be obtained for a video."""


def _snippet_text(snippet) -> str:
    # v1.x returns objects with .text, older versions returned dicts.
    return snippet["text"] if isinstance(snippet, dict) else snippet.text


def _join(snippets: Iterable) -> str:
    return " ".join(_snippet_text(s).replace("\n", " ").strip() for s in snippets)


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en", "hi")) -> str:
    """Return the transcript text for ``video_id``.

    Tries the preferred ``languages`` first (English, then Hindi). If neither exists,
    falls back to any available transcript and translates it to English when possible.
    """
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
        return _join(fetched)
    except TranscriptsDisabled as exc:
        raise TranscriptError("Captions are disabled for this video.") from exc
    except VideoUnavailable as exc:
        raise TranscriptError("This video is unavailable.") from exc
    except NoTranscriptFound:
        pass

    try:
        for transcript in api.list(video_id):
            if transcript.language_code != "en" and transcript.is_translatable:
                transcript = transcript.translate("en")
            return _join(transcript.fetch())
    except Exception as exc:  # network / parsing / blocked IP etc.
        raise TranscriptError(f"Could not fetch a transcript: {exc}") from exc

    raise TranscriptError("No transcript is available for this video.")
