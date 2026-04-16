from __future__ import annotations

from pathlib import Path
import sys


# Ensure imports work no matter where this script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.vector_store import build_vector_store


def main() -> None:
    manual_dir = Path("knowledge/manual")
    build_vector_store(manual_dir)
    print("知识库构建完成。")


if __name__ == "__main__":
    main()
