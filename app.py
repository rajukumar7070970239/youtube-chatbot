"""Streamlit web UI for the YouTube RAG chatbot."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ytchat.chain import YouTubeRAG

st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬")
st.title("🎬 YouTube Chatbot (RAG)")
st.caption("Ask questions about any YouTube video with captions, grounded in its transcript.")

url = st.text_input("YouTube URL or video ID")

if st.button("Load video") and url:
    with st.spinner("Fetching transcript and building the index..."):
        try:
            st.session_state.rag = YouTubeRAG(url)
            st.session_state.messages = []
            st.success(f"Loaded {len(st.session_state.rag.docs)} transcript chunks.")
        except Exception as exc:
            st.error(str(exc))

rag = st.session_state.get("rag")
if rag:
    if st.button("Summarize video"):
        with st.spinner("Summarizing..."):
            st.markdown(rag.summarize())

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about the video"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = rag.ask(question)
            st.markdown(answer)
            with st.expander("Retrieved context"):
                for doc in rag.retrieve(question):
                    st.write(f"**Chunk {doc.metadata.get('chunk')}**: {doc.page_content}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
