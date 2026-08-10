# AGENTS.md

> Instructions for Claude Code (and other AI coding agents) working on this project.

## Project Overview

This is a FastAPI server that exposes a `/layout-parsing` endpoint, making a vLLM-hosted PaddleOCR-VL model compatible with the PaddlePaddle AI Studio API format.

**Architecture**: Client → `/layout-parsing` → FastAPI → PaddleOCRVL (local PP-DocLayoutV3 layout detection + remote vLLM PaddleOCR-VL recognition)

## Key Files

| File | Purpose |
|---|---|
| `server.py` | Main FastAPI application — the only source file |
| `pyproject.toml` | Python project config and dependencies |
| `test_verify.py` | Verification script: compares local server vs AI Studio output |
| `AGENTS.md` | This file — agent instructions |
| `README.md` | Human-readable documentation |

## How to Run

```bash
# 1. Create venv and install deps
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2a. Option A — local pipeline (requires a vLLM server)
export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"

# 2b. Option B — AI Studio hosted API (no vLLM server needed)
# When AI_STUDIO_TOKEN is set, scanned PDFs and images are parsed via
# the paddleocr.aistudio hosted API instead of the local pipeline.
export AI_STUDIO_TOKEN="your-aistudio-access-token"

# 3. Start server
python server.py
# Server runs on port 8399 by default (set PORT env var to change)
```

## AI Studio Hosted API Mode

When `AI_STUDIO_TOKEN` is configured, the OCR route (scanned PDFs + images) in
`process_single_file()` is dispatched to `process_with_aistudio()` instead of
the local PaddleOCRVL pipeline — see the `AI_STUDIO_*` env vars at the top of
`server.py`. Text PDFs and Office files still go through local markitdown.
Relevant functions: `_build_aistudio_optional_payload`, `_merge_aistudio_results`,
`_resolve_image_value`, `process_with_aistudio`. Each JSONL line from the hosted
API carries one page's `result`, which is merged into the standard
`{layoutParsingResults, preprocessedImages, dataInfo}` response shape, with
remote image URLs downloaded and embedded as base64 data URLs.

## API Contract

### `POST /layout-parsing`

**Request body** (JSON):

```json
{
  "file": "<base64-encoded PDF or image>",
  "fileType": 0,
  "markdownIgnoreLabels": ["header", "footer", ...],
  "useLayoutDetection": true,
  "useSealRecognition": true,
  "mergeTables": true,
  "relevelTitles": true,
  "temperature": 0,
  "topP": 1,
  "repetitionPenalty": 1,
  "promptLabel": "ocr",
  "layoutNms": true,
  "restructurePages": true,
  ...
}
```

**Response** (JSON):

```json
{
  "logId": "uuid",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": { ... },
        "markdown": { "text": "...", "images": { "imgs/xxx.jpg": "data:..." } },
        "outputImages": { "layout_det_res": "data:..." },
        "inputImage": "data:..."
      }
    ],
    "preprocessedImages": ["data:..."],
    "dataInfo": { "type": "pdf", "numPages": N, "pages": [...] }
  }
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Parameter Mapping

AI Studio uses camelCase, PaddleOCRVL uses snake_case. The mapping is defined in `server.py` as `PARAM_MAPPING`. Key mappings:

| AI Studio | PaddleOCRVL |
|---|---|
| `markdownIgnoreLabels` | `markdown_ignore_labels` |
| `useLayoutDetection` | `use_layout_detection` |
| `useSealRecognition` | `use_seal_recognition` |
| `promptLabel` | `prompt_label` |
| `repetitionPenalty` | `repetition_penalty` |
| `topP` | `top_p` |

`mergeTables` and `relevelTitles` are passed to `pipeline.restructure_pages()` (post-processing).

## Important Notes

- PaddleOCRVL's result objects cannot be serialized with `dict()` — use `res.save_to_json()` and `res.save_to_markdown()` instead
- PDF files are first saved to a temp file because PaddleOCRVL needs a file path
- The PP-DocLayoutV3 model is downloaded automatically on first run (~125MB) to `~/.paddlex/official_models/`
- Images in responses use base64 data URLs (`data:image/jpeg;base64,...`)
- The vLLM server must already be running with PaddleOCR-VL model loaded

## Testing

```bash
# Run verification against AI Studio reference
python test_verify.py --pdf test.pdf
```

## Common Gotchas

1. **PaddlePaddle + vLLM conflict**: Use separate virtual environments if running both locally. The server only needs `paddleocr[doc-parser]` + `paddlepaddle` on the client side, and vLLM on the GPU server.
2. **PyMuPDF for PDF**: vLLM's PaddleOCR-VL only accepts images. PDF pages are rendered via PyMuPDF (`fitz`) at 150 DPI before processing.
3. **Layout detection runs locally**: PP-DocLayoutV3 (default for PaddleOCR-VL-1.5) runs on CPU by default. For GPU acceleration, install `paddlepaddle-gpu`.
