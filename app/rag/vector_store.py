from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable

from chromadb.config import Settings as ChromaClientSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import SETTINGS

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

_COLLECTION = "hvac_manual"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _chroma_client_settings() -> ChromaClientSettings:
    # Disable anonymized telemetry to avoid noisy runtime warnings.
    return ChromaClientSettings(anonymized_telemetry=False)


@lru_cache(maxsize=1)
def _embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_db() -> Chroma:
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_embedding(),
        persist_directory=str(SETTINGS.chroma_dir),
        client_settings=_chroma_client_settings(),
    )


def load_manual_documents(manual_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for file in manual_dir.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": file.name}))
    return docs


def _keyword_search_manual(query: str, top_k: int = 2) -> str:
    """Fallback retrieval that does not depend on embedding model downloads."""
    manual_dir = Path("knowledge/manual")
    if not manual_dir.exists():
        return "未检索到相关维修规程。"

    query_tokens = [t.strip() for t in query.replace("，", " ").replace("。", " ").split() if t.strip()]
    if not query_tokens:
        query_tokens = [query]

    scored: list[tuple[int, Path, str]] = []
    for file in manual_dir.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        score = sum(content.count(token) for token in query_tokens)
        scored.append((score, file, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = scored[:max(1, top_k)]

    lines: list[str] = []
    for idx, (_, file, content) in enumerate(top_items, start=1):
        lines.append(f"[{idx}] source={file.name}\n{content.strip()[:400]}")
    return "\n\n".join(lines) if lines else "未检索到相关维修规程。"


def build_vector_store(manual_dir: Path) -> Chroma:
    docs = load_manual_documents(manual_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    SETTINGS.chroma_dir.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding(),
        collection_name=_COLLECTION,
        persist_directory=str(SETTINGS.chroma_dir),
        client_settings=_chroma_client_settings(),
    )
    return db


def get_retriever(top_k: int = 2):
    db = _get_db()
    return db.as_retriever(search_kwargs={"k": top_k})


def search_manual(query: str, top_k: int = 2) -> str:
    try:
        retriever = get_retriever(top_k=top_k)
        docs = retriever.invoke(query)
    except Exception:
        return _keyword_search_manual(query=query, top_k=top_k)

    if not docs:
        return _keyword_search_manual(query=query, top_k=top_k)

    lines: list[str] = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        lines.append(f"[{idx}] source={source}\n{d.page_content.strip()[:400]}")
    return "\n\n".join(lines)
