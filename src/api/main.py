"""
FastAPI app for the code review summarizer.

Endpoints:
  GET  /             -> health / status
  POST /summarize_pr -> summarize a PR (via GitHub URL or manual input)

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Two RAG modes:
  - GitHub URL: fetches the changed files from that repo and builds
    a fresh index on the fly. Context is always relevant.
  - Manual input: uses the pre-built local index (run build_repo_index.py
    first if you want any retrieval).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must happen before LLMClient reads ANTHROPIC_API_KEY

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PRRequest, SummaryResponse
from src.github_client import GitHubClient
from src.llm.client import LLMClient, LLMConfig
from src.llm.summarizer import LLMPRSummarizer, PRSummaryInput
from src.rag.embedder import SimpleEmbedder
from src.rag.index import Chunk, SimpleIndex
from src.rag.retriever import PRRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/rag_index"))

# shared across requests
_embedder: SimpleEmbedder | None = None
_llm_client: LLMClient | None = None
_github: GitHubClient | None = None
_local_retriever: PRRetriever | None = None   # built from pre-indexed local repo
_llm_model_name: str = "unknown"


def _build_index_from_files(
    files: list[tuple[str, str]], chunk_size: int = 40
) -> SimpleIndex:
    """
    Build a fresh in-memory index from a list of (path, content) pairs.
    Used when we've fetched file contents from GitHub.
    """
    index = SimpleIndex()
    for path, content in files:
        lines = content.split("\n")
        for i, start in enumerate(range(0, len(lines), chunk_size)):
            end = min(start + chunk_size, len(lines))
            piece = "\n".join(lines[start:end]).strip()
            if not piece:
                continue
            chunk = Chunk(
                id=f"{path}:{i}",
                file_path=path,
                start_line=start + 1,
                end_line=end,
                text=piece,
            )
            emb = _embedder.embed_text(piece)
            index.add(chunk, emb)
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _llm_client, _github, _local_retriever, _llm_model_name

    _embedder = SimpleEmbedder()

    # try to load a pre-built local index for manual input mode
    local_index = SimpleIndex()
    if INDEX_DIR.exists() and (INDEX_DIR / "chunks.json").exists():
        logger.info(f"loading local RAG index from {INDEX_DIR}")
        local_index = SimpleIndex.load(INDEX_DIR)
        logger.info(f"  -> {len(local_index.chunks)} chunks loaded")
    else:
        logger.info(
            f"no local RAG index at {INDEX_DIR} "
            "(manual input will have no retrieval context - "
            "run scripts/build_repo_index.py to fix)"
        )
    _local_retriever = PRRetriever(local_index, _embedder)

    llm_cfg = LLMConfig()
    _llm_client = LLMClient(llm_cfg)
    _llm_model_name = llm_cfg.model if llm_cfg.provider != "mock" else "mock"

    _github = GitHubClient()

    logger.info(f"ready (llm={llm_cfg.provider}, model={_llm_model_name})")
    yield
    _embedder = _llm_client = _github = _local_retriever = None


app = FastAPI(
    title="Code Review Summarizer",
    description=(
        "Summarize a pull request by pasting a GitHub URL or raw diff. "
        "Uses RAG over the actual changed files and Claude to generate the summary."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": _llm_model_name,
        "ready": _llm_client is not None,
    }


@app.post("/summarize_pr", response_model=SummaryResponse)
def summarize_pr(req: PRRequest):
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="server not ready")

    source_url = None
    title = req.title
    description = req.description
    comments = req.comments
    diff_text = req.diff_text
    retriever = _local_retriever   # default: use local pre-built index

    if req.github_url:
        # fetch PR data + the actual files that were changed
        try:
            pr_data = _github.fetch(req.github_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.exception("GitHub fetch failed")
            raise HTTPException(status_code=502, detail=f"GitHub fetch failed: {e}")

        title = pr_data.title
        description = pr_data.description
        comments = pr_data.comments
        diff_text = pr_data.diff_text
        source_url = pr_data.source_url

        # build a fresh index from the files this PR actually touches
        # so retrieval is relevant to the repo being reviewed, not this project
        if pr_data.context_files:
            logger.info(f"building per-request index from {len(pr_data.context_files)} fetched files")
            fresh_index = _build_index_from_files(pr_data.context_files)
            retriever = PRRetriever(fresh_index, _embedder)
        else:
            logger.warning("no context files fetched - retrieval will be empty")

    pr = PRSummaryInput(
        title=title,
        description=description,
        comments=comments,
        diff_text=diff_text,
    )

    summarizer = LLMPRSummarizer(retriever, llm_client=_llm_client)

    try:
        result = summarizer.summarize(pr)
    except Exception as e:
        logger.exception("summarization failed")
        raise HTTPException(status_code=500, detail=str(e))

    return SummaryResponse(
        summary=result["summary"],
        title=title,
        source_url=source_url,
        retrieved_files=result["retrieved_files"],
        n_retrieved=result["num_retrieved"],
        model=_llm_model_name,
    )
