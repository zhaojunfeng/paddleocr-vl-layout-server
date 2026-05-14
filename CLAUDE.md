# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI server that exposes `/layout-parsing` and `/batch/layout-parsing` endpoints compatible with PaddlePaddle AI Studio, powered by PaddleOCR-VL + vLLM. Also supports markitdown for non-scanned documents and archive-based batch processing. Single-file architecture (`server.py`).

## Setup & Running

```bash
uv venv
uv pip install -e ".[dev]"
export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"
python server.py  # Starts on port 8399 (or PORT env var)
```

## Testing

```bash
# Compare local server output against AI Studio reference
python test_verify.py <file.pdf|file.png> [--aistudio]

# Deep field-by-field comparison with categorized diffs
python deep_compare.py <file.pdf|file.png> [--aistudio]
```

## Architecture

**Flow:** Client → This Server (:8399) → Route by file type:
- Images / scanned PDFs → PaddleOCRVL → PP-DocLayoutV3 (local CPU) + vLLM /v1/chat/completions (remote GPU)
- Non-scanned PDFs / DOCX / XLSX / HTML / etc. → markitdown (Microsoft) → markdown text

**Key design decisions:**
- PDF pages are rendered to PNG at 144 DPI via PyMuPDF before processing — PaddleOCRVL leaves `block_content` empty for PDF files directly
- Scanned PDF detection: first 2 pages with < 50 text characters = scanned (uses PyMuPDF text extraction)
- `PaddleOCRVL` result objects use `save_to_json()` / `save_to_markdown()` (not `dict()`)
- Parameter mapping: AI Studio camelCase → PaddleOCRVL snake_case via `PARAM_MAPPING` dict
- `mergeTables` / `relevelTitles` / `restructurePages` go to `pipeline.restructure_pages()`, not to `predict()`
- Images returned as base64 data URLs (`data:image/jpeg;base64,...`)
- Pipeline is lazy-initialized singleton via `get_pipeline()`
- `process_single_file()` supports both base64 (`file`+`fileType`) and server-side path (`filePath`) input
- Batch processing uses in-memory `BatchJobStore` with `ThreadPoolExecutor` (no external deps like Redis)
- Archive processing uses `ArchiveJobStore`, processes files via `process_single_file(filePath=...)`, saves results as zip

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_SERVER_URL` | `http://134.199.132.159/v1` | vLLM backend URL |
| `PORT` | `8399` | Server listen port |
| `MAX_BATCH_WORKERS` | `2` | Concurrent batch job threads |
| `JOB_TTL_SECONDS` | `3600` | Batch job expiry time |
| `SCANNED_PDF_CHAR_THRESHOLD` | `50` | Text char threshold for scanned PDF detection |
| `UPLOAD_DIR` | `./uploads` | Base directory for archive extraction and results |
| `ARCHIVE_RESULT_TTL_SECONDS` | `604800` (7 days) | Archive result zip expiry time |

## Common Gotchas

- PaddlePaddle and vLLM typically need separate virtual environments
- PP-DocLayoutV3 model auto-downloads on first run (~125MB)
- Layout detection runs on CPU by default
- `.json` files are git-ignored (except `pyproject.toml`)
- `markitdown[all]` pulls in converters for DOCX, XLSX, PPTX, etc. — lazy-imported to avoid startup overhead
- Archive extraction validates paths to prevent zip-slip attacks
- `UPLOAD_DIR` is created automatically at server startup
