## AI-Powered Code Review Pipeline

A side project where I'm building a system that:

- Takes a **pull request** (code diff + title + description + comments)
- Finds **related code context** from the repo using RAG
- Sends everything to **Claude** to generate a structured summary
- Exposes it all via a **FastAPI** backend and a **Streamlit** UI

It's intentionally not polished production SaaS. It's a realistic portfolio project
that shows how I think through system design, experiment, and structure code over time.

---

## What's in here

```
src/
  data/         data pipeline (processors, dataset)
  models/       multimodal model (ViT + CodeBERT + fusion)
  rag/          embedder, index, retriever
  llm/          Claude client, prompt builder, summarizer
  api/          FastAPI app

app.py          Streamlit UI
scripts/        CLI tools (build index, run pipeline, prepare data)
data/           raw + processed data, rag index (gitignored)
tests/          pytest suite
```

---

## Quick start

**1. Install deps**

```bash
pip install -r requirements.txt
```

**2. Set your API key**

```bash
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

**3. Build the RAG index** (indexes the repo's Python files)

```bash
python scripts/build_repo_index.py --repo-root . --out-dir data/rag_index
```

**4. Start the API server**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**5. Open the UI**

```bash
streamlit run app.py
```

Or call the API directly:

```bash
curl -X POST http://localhost:8000/summarize_pr \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fix null check in payment processor",
    "description": "Was crashing on empty cart",
    "comments": ["looks good", "add a test?"],
    "diff_text": "- if total:\n+ if total is not None:"
  }'
```

---

## Components

### Data pipeline (Phase 1 - done)

- `src/data/processors.py` - cleans git diffs, builds diff images (color-coded bars),
  combines PR context (title + description + comments)
- `src/data/dataset.py` - PyTorch Dataset wrapping processed PRs
- `scripts/prepare_data.py` - reads raw JSON, runs processors, splits train/val

### Multimodal model (Phase 2 - in progress)

The model looks at both the visual representation of a diff and the actual text,
then produces a fused representation that could be decoded into a summary.
Currently the summary head is a placeholder - the real summarization goes through
Claude via the RAG+LLM path below.

- `src/models/vision_transformer.py` - mini ViT for diff images (patches → CLS token)
- `src/models/code_bert.py` - BERT-style encoder for diff text + PR context
- `src/models/fusion.py` - cross-modal attention to mix image and text features
- `src/models/architecture.py` - ties everything together

### RAG + LLM pipeline (done)

This is the main working path right now:

1. Build an index over the repo's Python files (`scripts/build_repo_index.py`)
2. For a given PR, embed the query and retrieve the most relevant chunks
3. Build a prompt with the PR metadata + diff + retrieved chunks
4. Send to Claude, get back a bullet-point summary

- `src/rag/embedder.py` - sentence-transformers embeddings (all-MiniLM-L6-v2)
- `src/rag/index.py` - in-memory cosine similarity index (save/load to disk)
- `src/rag/retriever.py` - wraps the embedder + index into a retriever
- `src/llm/client.py` - Anthropic SDK wrapper (falls back to mock if no API key)
- `src/llm/prompt_builder.py` - assembles the full prompt
- `src/llm/summarizer.py` - orchestrates the whole flow

### API + UI

- `src/api/main.py` - FastAPI app with `POST /summarize_pr` and `GET /`
- `app.py` - Streamlit frontend

---

## Tech stack

- **LLM**: Claude (via Anthropic SDK)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **ML/DL**: PyTorch, custom transformer implementations
- **Image processing**: OpenCV, Pillow
- **API**: FastAPI + uvicorn
- **UI**: Streamlit
- **Data**: NumPy, pandas, scikit-learn
- **Config**: python-dotenv

---

## Running tests

```bash
pytest tests/ -v
```

The tests use mock mode for the LLM (no API key needed) and the hash-based
fallback for embeddings (no sentence-transformers download needed).

---

## What's next

- Swap the simple in-memory index for something like FAISS or ChromaDB
- Add caching so the same diff doesn't hit Claude twice (by commit SHA)
- Add a small eval script - compare generated summaries to human-written ones
- Maybe hook up to GitHub webhooks so it runs automatically on new PRs
