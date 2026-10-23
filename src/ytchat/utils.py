"""Small helpers that do not need network access."""

from __future__ import annotations

import re

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})"),
]


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video ID from a YouTube URL or a bare ID."""
    value = (url_or_id or "").strip()
    if _ID_RE.match(value):
        return value
    for pattern in _URL_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    raise ValueError(f"Could not find a YouTube video ID in: {url_or_id!r}")
