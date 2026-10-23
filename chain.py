"""Steps 4-6: retrieval, augmentation and generation, wired together as LangChain chains."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .chunking import split_transcript
from .config import Settings, get_settings
from .transcript import fetch_transcript
from .utils import extract_video_id
from .vectorstore import build_vector_store, get_embeddings, load_vector_store

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions about a YouTube video.
Answer ONLY from the transcript context below. If the context is not enough to
answer, say you don't know based on this video.

Context:
{context}

Question: {question}

Answer:"""
)

MAP_PROMPT = PromptTemplate.from_template(
    "Summarize the key points of this part of a video transcript in 3-5 concise "
    "bullet points:\n\n{text}"
)

REDUCE_PROMPT = PromptTemplate.from_template(
    "Below are bullet-point notes from consecutive parts of one YouTube video. "
    "Write a descriptive summary with an intro sentence, the main points as "
    "bullets, and a one-line takeaway.\n\n{notes}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class YouTubeRAG:
    """RAG pipeline: transcript -> chunks -> embeddings -> retriever -> prompt -> LLM."""

    def __init__(self, video: str, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        self.video_id = extract_video_id(video)
        self.llm = ChatOpenAI(
            model=self.settings.chat_model, api_key=self.settings.openai_api_key
        )
        self.embeddings = get_embeddings(
            self.settings.embedding_model, self.settings.openai_api_key
        )
        self.docs: list[Document] = []
        self.vector_store = None
        self.chain = None
        self._build()

    def _build(self) -> None:
        s = self.settings
        text = fetch_transcript(self.video_id)
        self.docs = split_transcript(text, self.video_id, s.chunk_size, s.chunk_overlap)

        self.vector_store = load_vector_store(
            self.embeddings, self.video_id, s.cache_dir
        ) or build_vector_store(self.docs, self.embeddings, self.video_id, s.cache_dir)

        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": s.top_k}
        )
        # Parallel branch: retrieve+format context while passing the question through.
        parallel = RunnableParallel(
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        )
        self.chain = parallel | QA_PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)

    def retrieve(self, question: str) -> list[Document]:
        return self.vector_store.similarity_search(question, k=self.settings.top_k)

    def summarize(self, group_size: int = 8) -> str:
        """Map-reduce summary over the whole transcript (retrieval alone would miss parts)."""
        map_chain = MAP_PROMPT | self.llm | StrOutputParser()
        groups = [
            format_docs(self.docs[i : i + group_size])
            for i in range(0, len(self.docs), group_size)
        ]
        notes = map_chain.batch([{"text": g} for g in groups])
        reduce_chain = REDUCE_PROMPT | self.llm | StrOutputParser()
        return reduce_chain.invoke({"notes": "\n\n".join(notes)})
