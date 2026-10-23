"""Command-line interface: chat with a YouTube video from the terminal."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse

from ytchat.chain import YouTubeRAG


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a YouTube video (RAG).")
    parser.add_argument("video", help="YouTube URL or 11-character video ID")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument("--summary", action="store_true", help="Print a summary and exit")
    args = parser.parse_args()

    try:
        rag = YouTubeRAG(args.video)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print(rag.summarize())
        return 0
    if args.question:
        print(rag.ask(args.question))
        return 0

    print(f"Loaded video {rag.video_id} ({len(rag.docs)} chunks). Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        print(f"\nBot: {rag.ask(question)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
