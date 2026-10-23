# 🎬 YouTube Chatbot – LangChain-based RAG System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about, and summarizes,
YouTube videos using their transcripts.

**Stack:** Python · LangChain · OpenAI · YouTube Transcript API · FAISS vector store · Streamlit

## How it works

```
YouTube URL ─▶ Transcript ─▶ Chunking ─▶ Embeddings ─▶ Vector store (FAISS)
                                                              │
 User question ─▶ Retriever (similarity top-k) ◀──────────────┘
        │                    │
        └──────▶ Grounded prompt (context + question) ─▶ LLM ─▶ Answer
```

| Phase | What happens | Code |
|-------|--------------|------|
| 1. Indexing – transcript | Fetch captions (English → Hindi → any, auto-translated to English) | `src/ytchat/transcript.py` |
| 2. Indexing – chunking | `RecursiveCharacterTextSplitter`, 1000 chars / 200 overlap | `src/ytchat/chunking.py` |
| 3. Indexing – embeddings | OpenAI embeddings stored in FAISS, cached per video | `src/ytchat/vectorstore.py` |
| 4. Retrieval | Similarity search returns top-k chunks | `src/ytchat/chain.py` |
| 5. Augmentation | Retrieved chunks + question merged into a grounded prompt | `src/ytchat/chain.py` |
| 6. Generation | LLM answers via a LangChain chain (`RunnableParallel | prompt | llm | parser`) | `src/ytchat/chain.py` |

Summaries use a map-reduce chain over all chunks so the whole video is covered.

## Quick start

```bash
git clone https://github.com/<your-username>/youtube-chatbot.git
cd youtube-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then add your OPENAI_API_KEY
```

### Web app
```bash
streamlit run app.py
```

### Command line
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"             # interactive chat
python main.py VIDEO_ID -q "What is the main topic?"                  # one question
python main.py VIDEO_ID --summary                                     # full summary
```

### Tests
```bash
pytest -q
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | – | Required |
| `CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking |
| `TOP_K` | `4` | Chunks retrieved per question |

## Limitations & ideas

- Needs a video with captions; some cloud IPs are blocked by YouTube.
- Planned: Chrome extension front-end, RAG evaluation (faithfulness, relevance, context recall)
  with LangSmith/RAGAS, semantic chunking, and swapping FAISS for Pinecone/Chroma.

## License

MIT
