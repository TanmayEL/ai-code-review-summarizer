"""
Builds a RAG index over a local repo's Python source files.

NOTE: You only need this for the "Manual Input" tab (when you're pasting
a raw diff without a GitHub URL). When you submit a GitHub URL, the API
automatically fetches the changed files from that repo and builds a
fresh index on the fly - no pre-built index needed.

Use this script when:
  - You're analyzing PRs for a private/local repo
  - You want to pre-index a specific codebase for faster manual analysis

Usage (from project root):
  python scripts/build_repo_index.py --repo-root . --out-dir data/rag_index
  python scripts/build_repo_index.py --repo-root /path/to/other/repo

First run downloads all-MiniLM-L6-v2 (~80MB). Subsequent runs are fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.rag.embedder import SimpleEmbedder
from src.rag.index import Chunk, SimpleIndex


def iter_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        # skip venv and the index script itself
        if "venv" in p.parts or ".git" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def chunk_file(path: Path, chunk_size: int = 40) -> List[Chunk]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = txt.split("\n")
    chunks: List[Chunk] = []

    start = 0
    idx = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        piece = "\n".join(lines[start:end]).strip()
        if piece:   # skip empty chunks
            chunks.append(
                Chunk(
                    id=f"{path}:{idx}",
                    file_path=str(path.as_posix()),
                    start_line=start + 1,
                    end_line=end,
                    text=piece,
                )
            )
        idx += 1
        start = end

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the RAG index for the code review summarizer"
    )
    parser.add_argument("--repo-root", type=str, default=".", help="root of the repo to index")
    parser.add_argument("--out-dir", type=str, default="data/rag_index", help="where to save the index")
    parser.add_argument("--chunk-size", type=int, default=40, help="lines per chunk")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning: {root}")
    py_files = iter_python_files(root)
    print(f"found {len(py_files)} python files")

    embedder = SimpleEmbedder()
    index = SimpleIndex()

    total_chunks = 0
    for f in py_files:
        chunks = chunk_file(f, chunk_size=args.chunk_size)
        for ch in chunks:
            emb = embedder.embed_text(ch.text)
            index.add(ch, emb)
        total_chunks += len(chunks)

    index.save(out_dir)
    print(f"saved index: {total_chunks} chunks -> {out_dir}")


if __name__ == "__main__":
    main()
