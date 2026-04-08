"""
Tests for the RAG + LLM pipeline.

These all run in offline mode:
- SimpleEmbedder falls back to hash-based embeddings if
  sentence-transformers isn't installed
- LLMClient uses provider="mock" so no API key is needed
"""

from pathlib import Path

import numpy as np

from src.llm.client import LLMClient, LLMConfig
from src.llm.summarizer import LLMPRSummarizer, PRSummaryInput
from src.rag.embedder import SimpleEmbedder
from src.rag.index import Chunk, SimpleIndex
from src.rag.retriever import PRInput, PRRetriever


def test_simple_embedder_shape():
    emb = SimpleEmbedder()
    v = emb.embed_text("some random text with a few tokens")
    assert isinstance(v, np.ndarray)
    # dim should match whatever the embedder reports
    assert v.shape[0] == emb.dim
    assert v.dtype == np.float32


def test_embedder_empty_input():
    emb = SimpleEmbedder()
    v = emb.embed_text("")
    assert v.shape[0] == emb.dim
    assert np.all(v == 0.0)


def test_index_add_and_search(tmp_path: Path):
    idx = SimpleIndex()
    emb = SimpleEmbedder()

    for i in range(3):
        ch = Chunk(
            id=f"c{i}",
            file_path=f"file_{i}.py",
            start_line=1,
            end_line=5,
            text=f"dummy code chunk {i}",
        )
        idx.add(ch, emb.embed_text(ch.text))

    q = emb.embed_text("dummy")
    results = idx.search(q, k=2)
    assert len(results) == 2

    chunk, score = results[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(score, float)

    # save + reload
    out_dir = tmp_path / "idx"
    idx.save(out_dir)
    loaded = SimpleIndex.load(out_dir)
    assert len(loaded.chunks) == len(idx.chunks)


def test_retriever_returns_results():
    idx = SimpleIndex()
    emb = SimpleEmbedder()

    ch = Chunk(
        id="c0",
        file_path="foo.py",
        start_line=1,
        end_line=10,
        text="def hello():\n    print('hi there')\n",
    )
    idx.add(ch, emb.embed_text(ch.text))

    retriever = PRRetriever(idx, emb)
    pr = PRInput(
        title="Add hello function",
        description="Adds a simple greeting function",
        comments=[],
        diff_text="+ def hello():\n+     print('hi there')",
    )
    results = retriever.retrieve(pr, k=3)
    assert len(results) >= 1
    assert isinstance(results[0][0], Chunk)


def test_llm_summarizer_mock_mode():
    idx = SimpleIndex()
    emb = SimpleEmbedder()

    ch = Chunk(
        id="c0",
        file_path="foo.py",
        start_line=1,
        end_line=5,
        text="def foo(x):\n    return x + 1\n",
    )
    idx.add(ch, emb.embed_text(ch.text))

    retriever = PRRetriever(idx, emb)
    # always mock - no API key needed
    llm = LLMClient(LLMConfig(provider="mock"))
    summarizer = LLMPRSummarizer(retriever, llm_client=llm)

    pr = PRSummaryInput(
        title="Add foo helper",
        description="Small helper to increment a value",
        comments=["looks fine", "maybe add tests later"],
        diff_text="+ def foo(x):\n+     return x + 1\n",
    )

    out = summarizer.summarize(pr)
    assert "summary" in out
    assert isinstance(out["summary"], str)
    assert len(out["summary"]) > 0
    assert isinstance(out["num_retrieved"], int)
    assert isinstance(out["retrieved_files"], list)
