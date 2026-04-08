"""
FastAPI app for the code review summarizer.

Endpoints:
  GET  /             -> health check / status
  POST /summarize_pr -> main endpoint, returns a PR summary

Run locally:
  uvicorn src.api.main:app --reload --port 8000

The RAG index needs to be built first:
  python scripts/build_repo_index.py --repo-root . --out-dir data/rag_index

If the index doesn't exist the API still works, retrieval just returns nothing.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PRRequest, SummaryResponse
from src.llm.client import LLMClient, LLMConfig
from src.llm.summarizer import LLMPRSummarizer, PRSummaryInput
from src.rag.embedder import SimpleEmbedder
from src.rag.index import SimpleIndex
from src.rag.retriever import PRRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/rag_index"))

# module-level state - loaded once at startup, reused for every request
_summarizer: LLMPRSummarizer | None = None
_llm_model_name: str = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _summarizer, _llm_model_name

    # load index
    index = SimpleIndex()
    if INDEX_DIR.exists() and (INDEX_DIR / "chunks.json").exists():
        logger.info(f"loading RAG index from {INDEX_DIR}")
        index = SimpleIndex.load(INDEX_DIR)
        logger.info(f"  -> {len(index.chunks)} chunks loaded")
    else:
        logger.warning(
            f"RAG index not found at {INDEX_DIR}. "
            "Retrieval will return nothing until you run: "
            "python scripts/build_repo_index.py"
        )

    embedder = SimpleEmbedder()
    retriever = PRRetriever(index, embedder)

    llm_cfg = LLMConfig()
    llm_client = LLMClient(llm_cfg)
    _llm_model_name = llm_cfg.model if llm_cfg.provider != "mock" else "mock"

    _summarizer = LLMPRSummarizer(retriever, llm_client=llm_client)
    logger.info(f"summarizer ready (provider={llm_cfg.provider})")

    yield

    _summarizer = None


app = FastAPI(
    title="Code Review Summarizer",
    description=(
        "RAG-powered PR summarization using Claude. "
        "Retrieves related code context from the repo, "
        "then asks Claude to write a structured summary."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# allow the Streamlit frontend to hit the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": _llm_model_name,
        "ready": _summarizer is not None,
    }


@app.post("/summarize_pr", response_model=SummaryResponse)
def summarize_pr(req: PRRequest):
    if _summarizer is None:
        raise HTTPException(status_code=503, detail="summarizer not ready")

    pr = PRSummaryInput(
        title=req.title,
        description=req.description,
        comments=req.comments,
        diff_text=req.diff_text,
    )

    try:
        result = _summarizer.summarize(pr)
    except Exception as e:
        logger.exception("summarization failed")
        raise HTTPException(status_code=500, detail=str(e))

    return SummaryResponse(
        summary=result["summary"],
        retrieved_files=result["retrieved_files"],
        n_retrieved=result["num_retrieved"],
        model=_llm_model_name,
    )
