# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local RAG (Retrieval-Augmented Generation) knowledge base Q&A system. Documents are stored in `library/`, indexed into ChromaDB (vector search) + BM25 (keyword search), and queried via hybrid retrieval with optional reranking. The LLM layer is swappable via a provider abstraction supporting Ollama (local) and any OpenAI-compatible API.

## Running the App

**Web UI (primary interface):**
```bash
python api.py
# or
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Opens at `http://127.0.0.1:8000`. The FastAPI app serves the frontend from `static/` and auto-indexes `library/` on startup.

**CLI (legacy, non-streaming):**
```bash
python main.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Configuration

`config.json` is gitignored and never committed. On first run it is auto-created from `config.example.json`. Edit `config.json` directly or use the web UI settings panel.

The default config uses local Ollama (`qwen2.5:7b`). To add an OpenAI-compatible provider, add a new entry under `providers` with `"type": "openai"`, `"base_url"`, and `"api_key"`.

## Architecture

### Retrieval Pipeline (`retriever.py`)
`retrieve(question, db_path, scope)` runs three stages:
1. **Vector search** via ChromaDB (`BAAI/bge-small-zh-v1.5` embeddings)
2. **BM25 keyword search** via `rank_bm25` (corpus cached as `db_bm25.pkl`)
3. **RRF fusion** merges both ranked lists, then optional **CrossEncoder reranking** (`BAAI/bge-reranker-base`, ~280 MB, off by default until downloaded)

`scope` filters results to a folder prefix or specific file path within `library/`.

### Indexing (`indexer.py`)
`build_index(library_folder, db_path, meta_path, force, progress_cb)` does **incremental indexing** — only files whose mtime changed are re-indexed. File removals are also synced. Both ChromaDB and BM25 corpus are updated atomically.

Chunking strategy (`CHUNK_VERSION = "semantic_v1"`): `.md` files split on `#`/`##`/`###` headings; other formats split on blank lines; oversized chunks recurse into sentence-level splits then sliding-window fallback. Incrementing `CHUNK_VERSION` triggers automatic full rebuild on next startup.

### Provider Abstraction (`providers/`)
`ChatProvider` (ABC) exposes `stream_chat(messages) -> Iterator[str]` and `test() -> (bool, str)`.
- `OllamaProvider` — local Ollama via the `ollama` Python SDK
- `OpenAIProvider` — any OpenAI-compatible endpoint (DeepSeek, Kimi, OpenAI, vLLM, etc.)

`build_provider(profile_dict)` is the factory; `config.py` manages reading/writing `config.json` and constructing the active provider.

### Streaming Chat (`chat.py` → `api.py`)
`ask_stream()` yields SSE events: `sources` → `loading` → `chunk`... → `done` | `error`.
`api.py` wraps this as `POST /api/chat/stream` (Server-Sent Events). Conversation history (last 12 messages) is loaded from `history/{conv_id}.json` and injected into the prompt.

### Background Jobs (`api.py`)
Upload and rebuild operations return a `job_id` immediately. The frontend polls `GET /api/jobs/{job_id}` for progress. Job state lives in the in-process `_jobs` dict (lost on restart).

## Key Paths

| Path | Purpose |
|------|---------|
| `library/` | Source documents (PDF, DOCX, MD); gitignored |
| `db/` | ChromaDB persistent store; gitignored |
| `db_bm25.pkl` | BM25 corpus pickle; gitignored |
| `db_meta.json` | Incremental index metadata (mtime + chunk IDs per file); gitignored |
| `history/` | Per-conversation JSON files; gitignored |
| `hf-cache/` | HuggingFace model cache (bge-small-zh + optional reranker) |
| `static/` | Frontend HTML/JS served by FastAPI |

## Adding a New LLM Provider

1. Create `providers/your_provider.py` implementing `ChatProvider.stream_chat()`.
2. Register it in `providers/__init__.py` and add a branch in `build_provider()`.
3. Add a sample entry in `config.example.json`.
