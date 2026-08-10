# PaddleOCR-VL Layout Parsing Server

A FastAPI server that adds the `/layout-parsing` endpoint to any vLLM instance running PaddleOCR-VL, making it fully compatible with the [PaddlePaddle AI Studio](https://aistudio.baidu.com/) API format.

## Why This Exists

AMD GPU servers running PaddleOCR-VL via vLLM only expose the OpenAI-compatible `/v1/chat/completions` endpoint. However, the standard PaddleOCR client code uses the `/layout-parsing` endpoint with a specific request/response format (PDF support, layout detection, structured Markdown output, etc.).

This server bridges that gap — deploy it alongside vLLM to get full AI Studio compatibility.

## Architecture

```
                    AMD GPU Server
                 ┌─────────────────────────┐
                 │                         │
  Client ──────►│  This Server (:8399)    │
  (AI Studio    │     └──► PaddleOCRVL   │
   format)      │           ├─ PP-DocLayoutV3 (local, CPU)     │
                │           └─ vLLM /v1/chat/completions (GPU)  │
                │                         │
                └─────────────────────────┘
```

- **Layout Detection** (PP-DocLayoutV3): Runs locally on CPU, identifies document elements (titles, paragraphs, tables, formulas, images, etc.)
- **Element Recognition** (PaddleOCR-VL-0.9B): Runs on the remote vLLM server (AMD GPU), performs OCR/table/formula/chart recognition

## Quick Start

### Prerequisites

- Python 3.10+
- A running vLLM server with PaddleOCR-VL model loaded (see [vLLM docs](https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html))

### Install

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run

```bash
# Set the vLLM server URL
export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"

# Start the server (default port: 8399)
python server.py

# Or with custom port
PORT=9000 python server.py
```

### AI Studio Hosted API Mode (no local vLLM needed)

If you have an [AI Studio access token](https://aistudio.baidu.com/), you can skip the local vLLM deployment entirely — scanned PDFs and images are sent to the `paddleocr.aistudio` hosted API instead of the local PaddleOCRVL + vLLM pipeline:

```bash
# Configure the token (the hosted API is then used automatically)
export AI_STUDIO_TOKEN="your-aistudio-access-token"
export AI_STUDIO_MODEL="PaddleOCR-VL-1.6"   # optional, default: PaddleOCR-VL-1.6
python server.py
```

When `AI_STUDIO_TOKEN` is set:

- Scanned PDFs and images (`/layout-parsing`, batch, and archive jobs) are parsed via the hosted API.
- Text PDFs and Office files still go through local markitdown.
- The local vLLM server is **not** contacted; `VLLM_SERVER_URL` can be left unset.
- Results are returned in the exact same format as the local pipeline (remote image URLs are downloaded and embedded as base64 data URLs).

| Env var | Default | Description |
|---|---|---|
| `AI_STUDIO_TOKEN` | *(unset)* | Access token; when set, enables the hosted API mode |
| `AI_STUDIO_MODEL` | `PaddleOCR-VL-1.6` | Model name used for hosted jobs |
| `AI_STUDIO_JOB_URL` | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | Job API base URL |
| `AI_STUDIO_POLL_INTERVAL` | `5` | Poll interval in seconds |
| `AI_STUDIO_TIMEOUT` | `1800` | Max wait for a job (seconds) |
| `AI_STUDIO_REQUEST_TIMEOUT` | `120` | Per-HTTP-request timeout (seconds) |

### Test

```bash
# Health check
curl http://localhost:8399/health

# Process a PDF
python -c "
import base64, requests, json

with open('test.pdf', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post('http://localhost:8399/layout-parsing', json={
    'file': b64,
    'fileType': 0,
    'useLayoutDetection': True,
    'temperature': 0,
})

data = resp.json()
for i, page in enumerate(data['result']['layoutParsingResults']):
    print(f'Page {i}: {len(page[\"markdown\"][\"text\"])} chars')
"
```

## API Reference

### `POST /layout-parsing`

Process a document (PDF or image) and return structured layout parsing results.

**Request Body:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | string | **required** | Base64-encoded file content |
| `fileType` | int | **required** | `0` = PDF, `1` = image |
| `markdownIgnoreLabels` | string[] | `null` | Layout labels to exclude from markdown (e.g., `"header"`, `"footer"`) |
| `useLayoutDetection` | bool | `true` | Enable layout detection |
| `useSealRecognition` | bool | `true` | Enable seal/stamp recognition |
| `useChartRecognition` | bool | `false` | Enable chart recognition |
| `useOcrForImageBlock` | bool | `false` | Use OCR for image blocks |
| `mergeTables` | bool | `true` | Merge multi-page tables |
| `relevelTitles` | bool | `true` | Re-level heading hierarchy |
| `restructurePages` | bool | `true` | Restructure page content |
| `layoutNms` | bool | `true` | Apply NMS to layout detection |
| `promptLabel` | string | `"ocr"` | Recognition prompt (`"ocr"`, `"table"`, `"formula"`, `"chart"`) |
| `temperature` | float | `0` | Generation temperature |
| `topP` | float | `1` | Top-p sampling |
| `repetitionPenalty` | float | `1` | Repetition penalty |
| `minPixels` | int | `147384` | Min pixels for image processing |
| `maxPixels` | int | `2822400` | Max pixels for image processing |
| `layoutShapeMode` | string | `"auto"` | Layout polygon mode |

**Response:**

```json
{
  "logId": "uuid",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": {
          "page_count": 1,
          "width": 1224,
          "height": 1584,
          "model_settings": { ... },
          "parsing_res_list": [
            {
              "block_label": "text",
              "block_content": "...",
              "block_bbox": [x1, y1, x2, y2],
              "block_id": 0,
              "block_order": 0,
              "group_id": 0,
              "block_polygon_points": [[x, y], ...]
            }
          ],
          "layout_det_res": {
            "boxes": [
              {
                "cls_id": 0,
                "label": "text",
                "score": 0.98,
                "coordinate": [x1, y1, x2, y2],
                "order": 0
              }
            ]
          }
        },
        "markdown": {
          "text": "# Title\n\nParagraph content...",
          "images": {
            "imgs/img_in_image_box_xxx.jpg": "data:image/jpeg;base64,..."
          }
        },
        "outputImages": {
          "layout_det_res": "data:image/jpeg;base64,..."
        },
        "inputImage": "data:image/jpeg;base64,..."
      }
    ],
    "preprocessedImages": ["data:image/jpeg;base64,..."],
    "dataInfo": {
      "type": "pdf",
      "numPages": 15,
      "pages": [{"width": 1224, "height": 1584}, ...]
    }
  }
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## How It Works

1. **Receive** base64-encoded file via `/layout-parsing`
2. **Decode** and save to temp file (PaddleOCRVL needs a file path)
3. **Layout Detection** — PP-DocLayoutV3 (default for PaddleOCR-VL-1.5) runs locally to identify document structure
4. **Element Recognition** — Each detected block is sent to the vLLM server for OCR/table/formula recognition
5. **Post-processing** — `restructure_pages()` merges tables, re-levels headings across pages
6. **Format** — Convert results to AI Studio response format with Markdown, images, and structured JSON

## Deployment for AMD

### Recommended Setup

```bash
# On the AMD GPU server, start vLLM first:
vllm serve PaddlePaddle/PaddleOCR-VL \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching

# Then start this server (can be on same machine or different):
VLLM_SERVER_URL="http://localhost:8000/v1" python server.py
```

### Docker (Optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
ENV VLLM_SERVER_URL=http://vllm:8000/v1
EXPOSE 8399
CMD ["python", "server.py"]
```

## Docker Deployment

项目自带 `Dockerfile`、`docker-compose.yaml` 与 `.env.example`，支持两种运行模式。

### 1. 准备环境变量

```bash
cp .env.example .env
# 编辑 .env：二选一填写
#   - AI Studio 托管模式：AI_STUDIO_TOKEN=xxx（无需 vLLM）
#   - 本地 vLLM 模式：VLLM_SERVER_URL=http://your-vllm:8000/v1
```

### 2. 启动

```bash
# 仅启动本服务（默认；连接外部 vLLM 或 AI Studio 托管）
docker compose up -d --build

# 连同内置 vLLM（GPU）一起启动
docker compose --profile vllm up -d --build
```

### 3. 获取访问令牌

**所有接口都在 JWT 鉴权之下（包括 `/health`）**，管理员 token 每次启动打印到日志：

```bash
docker compose logs layout-server | grep "Admin token"
# [auth] Admin token: eyJhbGciOi...
```

调用示例：

```bash
curl http://localhost:8399/layout-parsing \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"file": "<base64>", "fileType": 0}'
```

### 说明

- **健康检查**：容器内采用 TCP 探测（`/health` 也在鉴权之下，无法无 token 探测）。
- **持久化**：`layout-data` 卷（uploads / 任务 / user.json）与 `paddlex-models` 卷（Paddle 模型缓存，首次本地模式运行下载 ~125MB）。
- **重启 token 失效**：若修改了 `JWT_SECRET`，旧 token 全部失效；建议 `.env` 固定一个随机值。
- **内置 vLLM**：需要 NVIDIA GPU + `nvidia-container-toolkit`，模型名/参数可按实际环境修改 compose 中 `vllm` 服务。

## Verified Compatibility

This server has been verified against the PaddlePaddle AI Studio `/layout-parsing` endpoint using identical input (`1.pdf`, 15 pages, full parameter set) and a field-by-field deep comparison.

### Test method

Run the included `deep_compare.py` script, which calls both AI Studio and the local server with the exact same payload, then compares every JSON field across all 15 pages:

```bash
# 1. Start the server
VLLM_SERVER_URL="http://your-vllm:8000/v1" python server.py

# 2. Run deep comparison
python deep_compare.py path/to/test.pdf --aistudio
# AI Studio token can be set via AISTUDIO_TOKEN env var
```

### Full-parameter test payload

The test uses the complete AI Studio parameter set:

```json
{
  "file": "<base64>",
  "fileType": 0,
  "markdownIgnoreLabels": ["header","header_image","footer","footer_image","number","footnote","aside_text"],
  "useDocOrientationClassify": false,
  "useDocUnwarping": false,
  "useLayoutDetection": true,
  "useChartRecognition": false,
  "useSealRecognition": true,
  "useOcrForImageBlock": false,
  "mergeTables": true,
  "relevelTitles": true,
  "layoutShapeMode": "auto",
  "promptLabel": "ocr",
  "repetitionPenalty": 1,
  "temperature": 0,
  "topP": 1,
  "minPixels": 147384,
  "maxPixels": 2822400,
  "layoutNms": true,
  "restructurePages": true
}
```

### Results (1.pdf, 15 pages)

```
Field-by-Field Deep Comparison: AI Studio vs Local Server
══════════════════════════════════════════════════════════

  Top-level:
    errorCode       : 0 == 0             PASS
    errorMsg        : Success == Success  PASS

  dataInfo:
    type            : pdf == pdf          PASS
    numPages        : 15 == 15            PASS
    pages[].width   : all match           PASS
    pages[].height  : all match           PASS

  Per-page (x15 pages):
    prunedResult keys                    PASS
    model_settings keys & values        PASS
    parsing_res_list block labels        PASS
    parsing_res_list block IDs           PASS
    parsing_res_list block orders        PASS
    parsing_res_list block groups        PASS
    layout_det_res structure             PASS
    markdown.text present                PASS
    markdown.images structure            PASS
    outputImages keys                    PASS
    inputImage present                   PASS

  Summary:
    Critical    (structure/format) :  0   PASS
    Structural  (missing fields)   :  0   PASS
    Content     (text differences) : ~40  (see note below)
    Precision   (bbox/score)       : ~19  (see note below)
    Other       (image naming)     :  ~7  (see note below)
```

### Interpretation of remaining differences

| Category | Count | Cause | Impact |
|---|---|---|---|
| **Precision** | ~19 | bbox coordinates ±1-3px, detection score ±0.05, polygon vertex offsets | None — layout detection runs independently on each backend |
| **Content** | ~40 | Some blocks have different or empty `block_content`; markdown text length varies across pages | None — the 0.9B VLM model produces non-identical output across different inference backends (PaddlePaddle vs vLLM/PyTorch) even with `temperature=0` |
| **Other** | ~7 | Image filenames differ by 1px in coordinate (e.g., `img_in_image_box_390_141_833_786.jpg` vs `785.jpg`); occasional box count ±1 | None — different layout detection runs produce slightly different bounding boxes |

**These differences are inherent to running the same model on different inference frameworks.** They do not affect API compatibility — any client code written for AI Studio will work without modification against this server.

### Key design decision: PDF → Image conversion

PaddleOCRVL leaves `block_content` empty for many layout blocks when processing PDF files directly. AI Studio renders PDFs to images internally before processing. This server does the same — each PDF page is rendered to a PNG image at 144 DPI via PyMuPDF before being passed to PaddleOCRVL. This ensures `block_content` is populated correctly and `dataInfo.pages` dimensions match AI Studio's output.

## License

MIT
