from __future__ import annotations

from app.rag.vector_store import search_manual


def search_manual_tool(question: str) -> str:
    """检索客室空调维修手册并返回可引用文本。"""
    return search_manual(question, top_k=2)
