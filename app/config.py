from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    ollama_num_gpu: int = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./.chroma"))
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


SETTINGS = Settings()
