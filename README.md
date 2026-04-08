# AI-Powered Code Review Pipeline

I built this because code review is genuinely painful at scale. You get a 400-line diff, no context, and you are expected to understand what changed, why it changed, and whether it's safe to merge. This tool tries to automate the first part of that: reading the PR, pulling in the surrounding code context, and giving you a structured summary you would actually use.

Paste any public GitHub PR or commit URL. It fetches the diff, grabs the full content of the files that changed, runs retrieval over them to find the most relevant chunks, and sends everything to Claude. You get a bullet-point summary in a few seconds.

---

> **[screenshot — main UI, GitHub URL tab]**

---

## What it does

- **Fetches PR data automatically** from GitHub: title, description, comments, and the full diff. No copy-pasting.
- **Retrieves actual file context**: parses the diff to find which files changed, fetches their full content from GitHub at the base commit, and builds a local vector index on the fly. Claude sees the whole file, not just the lines that changed.
- **Summarizes with Claude**: sends the PR metadata + diff + retrieved context in a structured prompt. Output is 3–6 bullet points covering what changed, why, and anything worth flagging in review.
- **Two ways to use it**: paste a GitHub URL (easiest), or use the Manual Input tab for private repos or local diffs.

---

> **[screenshot — summary output with retrieved context files]**

---

## How it works

```
GitHub PR URL
      ↓
  GitHubClient
  ├── fetch PR title, description, comments
  ├── fetch the diff
  ├── parse which files changed from the diff
  └── fetch full content of those files at base commit
      ↓
  Build per-request RAG index
  ├── chunk each file into ~40-line pieces
  ├── embed with all-MiniLM-L6-v2 (sentence-transformers)
  └── cosine similarity search → top-5 relevant chunks
      ↓
  Build prompt
  ├── system instruction
  ├── PR metadata (title, description, comments)
  ├── diff (truncated if huge)
  └── retrieved code chunks with file + line references
      ↓
  Claude (claude-opus-4-6)
      ↓
  Structured bullet-point summary
```

The key design decision: the RAG index is built fresh per request from the actual files in the PR's repo — not from some pre-built index of an unrelated codebase. So context is always relevant.

---

## Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/) (required)
- A [GitHub token](https://github.com/settings/tokens) (optional - you get 60 unauthenticated requests/hour, which is fine for occasional use)

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/TanmayEL/multimodal-code-summarizer
cd multimodal-code-summarizer
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
```

Open `.env` and set your keys:

```
ANTHROPIC_API_KEY=sk-ant-...
GITHUB_TOKEN=ghp_...        # optional
```

```bash
# 3. Start the API (terminal 1)
uvicorn src.api.main:app --reload --port 8000

# 4. Start the UI (terminal 2)
streamlit run app.py
```

Open **http://localhost:8501**, paste a GitHub PR URL, hit **Summarize**.

---

<!-- Add screenshot: the sidebar with API health check showing "Online | model: claude-opus-4-6" -->

> **[screenshot — sidebar showing API online + model name]**

---

## Using the API directly

The FastAPI server exposes a single endpoint. You can call it without the UI:

```bash
# from a GitHub URL
curl -X POST http://localhost:8000/summarize_pr \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/django/django/pull/1234"}'

# or paste a raw diff manually
curl -X POST http://localhost:8000/summarize_pr \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fix off-by-one in pagination",
    "description": "Page count was wrong when total items % page_size == 0",
    "diff_text": "- pages = total // page_size\n+ pages = math.ceil(total / page_size)"
  }'
```

Response shape:

```json
{
  "summary": "• Fixed off-by-one bug in pagination...",
  "title": "Fix off-by-one in pagination",
  "source_url": "https://github.com/...",
  "retrieved_files": ["src/pagination.py"],
  "n_retrieved": 5,
  "model": "claude-opus-4-6"
}
```

Swagger docs at **http://localhost:8000/docs**.

---

## For private repos / local diffs

Use the **Manual Input** tab — paste your diff, title, and description directly. If you want retrieval context for a private repo, pre-build an index from it:

```bash
python scripts/build_repo_index.py --repo-root /path/to/your/repo
```

This indexes all `.py` files and saves to `data/rag_index/`. The API picks it up automatically on restart.

---

## Project structure

```
src/
  github_client.py    fetch PR/commit data + changed file contents from GitHub
  rag/                embedder (sentence-transformers), vector index, retriever
  llm/                Claude client, prompt builder, summarizer orchestrator
  api/                FastAPI app + Pydantic schemas
  models/             multimodal model — ViT + CodeBERT + fusion (Phase 2, WIP)
  data/               data processors + PyTorch dataset

app.py                Streamlit UI
scripts/              CLI tools (build index, prepare data, run pipeline)
tests/                pytest suite (runs fully offline, no API key needed)
```

---

## Tech stack

| Layer            | What                                                 |
| ---------------- | ---------------------------------------------------- |
| LLM              | Claude (`claude-opus-4-6`) via Anthropic SDK         |
| Embeddings       | `all-MiniLM-L6-v2` via sentence-transformers         |
| Vector search    | In-memory cosine similarity (NumPy)                  |
| API              | FastAPI + uvicorn                                    |
| UI               | Streamlit                                            |
| ML (Phase 2)     | PyTorch — custom ViT + CodeBERT + cross-modal fusion |
| Image processing | OpenCV + Pillow                                      |
