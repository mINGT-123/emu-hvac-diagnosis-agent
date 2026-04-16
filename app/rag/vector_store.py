from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import SETTINGS

_COLLECTION = "hvac_manual"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_db() -> Chroma:
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_embedding(),
        persist_directory=str(SETTINGS.chroma_dir),
    )


def load_manual_documents(manual_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for file in manual_dir.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": file.name}))
    return docs


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
    )
    db.persist()
    return db


def get_retriever(top_k: int = 2):
    db = _get_db()
    return db.as_retriever(search_kwargs={"k": top_k})


def search_manual(query: str, top_k: int = 2) -> str:
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(query)

    if not docs:
        return "未检索到相关维修规程。"

    lines: list[str] = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        lines.append(f"[{idx}] source={source}\n{d.page_content.strip()[:400]}")
    return "\n\n".join(lines)
