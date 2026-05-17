"""
PaddleOCR-VL Layout Parsing Server
====================================
A FastAPI server that wraps PaddleOCRVL + vLLM backend to expose
a /layout-parsing endpoint compatible with PaddlePaddle AI Studio.

Architecture:
  Client -> /layout-parsing -> FastAPI -> PaddleOCRVL (local layout + remote vLLM)

Usage:
  # Set vLLM server URL (default: http://localhost:8000/v1)
  export VLLM_SERVER_URL="http://your-vllm-server:8000/v1"
  python server.py
"""

import asyncio
import base64
import copy
import hashlib
import io
import json
import os
import secrets
import shutil
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import cv2
import jwt
import numpy as np
from fastapi import Depends, FastAPI, Header, Request, UploadFile, File as FastAPIFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

# ─── Configuration ───────────────────────────────────────────────────
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://134.199.132.159/v1")
SERVER_PORT = int(os.environ.get("PORT", "8399"))
MAX_BATCH_WORKERS = int(os.environ.get("MAX_BATCH_WORKERS", "2"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))
SCANNED_PDF_CHAR_THRESHOLD = int(os.environ.get("SCANNED_PDF_CHAR_THRESHOLD", "50"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
JOBS_DIR = os.environ.get("JOBS_DIR", os.path.join(UPLOAD_DIR, "jobs"))
ARCHIVE_RESULT_TTL_SECONDS = int(os.environ.get("ARCHIVE_RESULT_TTL_SECONDS", str(7 * 24 * 3600)))
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
USER_DB_PATH = os.environ.get("USER_DB_PATH", "user.json")
ADMIN_EMAIL = "admin@ionestep.com"

# ─── PaddleOCRVL lazy init ──────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from paddleocr import PaddleOCRVL
        _pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=VLLM_SERVER_URL,
        )
    return _pipeline


# ─── Auth ─────────────────────────────────────────────────────────

_user_db_lock = threading.Lock()


def _load_users() -> dict:
    if not os.path.exists(USER_DB_PATH):
        return {}
    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _ensure_admin():
    users = _load_users()
    if ADMIN_EMAIL not in users:
        users[ADMIN_EMAIL] = {
            "salt": secrets.token_hex(16),
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_users(users)
        print(f"[auth] Created admin user: {ADMIN_EMAIL}")
    # Always print admin token at startup for convenience
    token, _ = create_token(ADMIN_EMAIL)
    print(f"[auth] Admin token: {token}")


def _derive_key(salt: str) -> bytes:
    return hashlib.sha256((JWT_SECRET + salt).encode()).digest()


def create_token(email: str, expires_in: int | None = None) -> tuple[str, str | None]:
    """Create JWT for user. Returns (token, expires_at_iso_or_None)."""
    users = _load_users()
    if email not in users:
        raise ValueError(f"User not found: {email}")
    salt = users[email]["salt"]
    payload = {"email": email, "salt": salt, "iat": datetime.now(timezone.utc)}
    expires_at = None
    if expires_in is not None and expires_in > 0:
        exp_time = datetime.now(timezone.utc).timestamp() + expires_in
        payload["exp"] = exp_time
        expires_at = datetime.fromtimestamp(exp_time, tz=timezone.utc).isoformat()
    token = jwt.encode(payload, _derive_key(salt), algorithm="HS256")
    return token, expires_at


async def verify_token(request: Request, authorization: str = Header(None)):
    if not authorization:
        return JSONResponse(status_code=401, content={"error": "Missing Authorization header"})
    token = authorization.removeprefix("Bearer ").strip()
    # Decode without verification to get email
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.DecodeError:
        return JSONResponse(status_code=401, content={"error": "Invalid token format"})
    email = unverified.get("email")
    token_salt = unverified.get("salt")
    if not email or not token_salt:
        return JSONResponse(status_code=401, content={"error": "Missing email or salt in token"})
    # Check user exists and salt matches
    users = _load_users()
    if email not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if users[email]["salt"] != token_salt:
        return JSONResponse(status_code=401, content={"error": "Token revoked"})
    # Verify signature and expiry
    try:
        jwt.decode(token, _derive_key(token_salt), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return JSONResponse(status_code=401, content={"error": "Token expired"})
    except jwt.InvalidSignatureError:
        return JSONResponse(status_code=403, content={"error": "Invalid signature"})
    except jwt.DecodeError:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    # Attach email to request state for admin endpoints
    request.state.user_email = email


def _require_admin(request: Request):
    email = getattr(request.state, "user_email", None)
    if email != ADMIN_EMAIL:
        return JSONResponse(status_code=403, content={"error": "Admin access required"})


# ─── Request / Response Models ──────────────────────────────────────

class LayoutParsingRequest(BaseModel):
    file: Optional[str] = Field(None, description="Base64-encoded file content")
    fileType: Optional[int] = Field(None, description="0=PDF, 1=image (auto-detected if filePath provided)")
    filePath: Optional[str] = Field(None, description="Server-side file path (alternative to base64)")
    markdownIgnoreLabels: Optional[list[str]] = None
    useDocOrientationClassify: Optional[bool] = False
    useDocUnwarping: Optional[bool] = False
    useLayoutDetection: Optional[bool] = True
    useChartRecognition: Optional[bool] = False
    useSealRecognition: Optional[bool] = True
    useOcrForImageBlock: Optional[bool] = False
    mergeTables: Optional[bool] = True
    relevelTitles: Optional[bool] = True
    layoutShapeMode: Optional[str] = "auto"
    promptLabel: Optional[str] = "ocr"
    repetitionPenalty: Optional[float] = 1
    temperature: Optional[float] = 0
    topP: Optional[float] = 1
    minPixels: Optional[int] = 147384
    maxPixels: Optional[int] = 2822400
    layoutNms: Optional[bool] = True
    restructurePages: Optional[bool] = True

    @model_validator(mode="after")
    def validate_file_source(self):
        if not self.file and not self.filePath:
            raise ValueError("Either 'file' (base64) or 'filePath' must be provided")
        return self


class BatchFileRequest(BaseModel):
    file: Optional[str] = Field(None, description="Base64-encoded file content")
    fileType: Optional[int] = Field(None, description="0=PDF, 1=image (auto-detected if filePath provided)")
    filePath: Optional[str] = Field(None, description="Server-side file path (alternative to base64)")
    markdownIgnoreLabels: Optional[list[str]] = None
    useDocOrientationClassify: Optional[bool] = False
    useDocUnwarping: Optional[bool] = False
    useLayoutDetection: Optional[bool] = True
    useChartRecognition: Optional[bool] = False
    useSealRecognition: Optional[bool] = True
    useOcrForImageBlock: Optional[bool] = False
    mergeTables: Optional[bool] = True
    relevelTitles: Optional[bool] = True
    layoutShapeMode: Optional[str] = "auto"
    promptLabel: Optional[str] = "ocr"
    repetitionPenalty: Optional[float] = 1
    temperature: Optional[float] = 0
    topP: Optional[float] = 1
    minPixels: Optional[int] = 147384
    maxPixels: Optional[int] = 2822400
    layoutNms: Optional[bool] = True
    restructurePages: Optional[bool] = True

    @model_validator(mode="after")
    def validate_file_source(self):
        if not self.file and not self.filePath:
            raise ValueError("Either 'file' (base64) or 'filePath' must be provided")
        return self


class BatchSubmitRequest(BaseModel):
    files: list[BatchFileRequest] = Field(..., min_length=1, max_length=50)
    callbackUrl: Optional[str] = None


# ─── Parameter Mapping ──────────────────────────────────────────────

PARAM_MAPPING = {
    "markdownIgnoreLabels": "markdown_ignore_labels",
    "useDocOrientationClassify": "use_doc_orientation_classify",
    "useDocUnwarping": "use_doc_unwarping",
    "useLayoutDetection": "use_layout_detection",
    "useChartRecognition": "use_chart_recognition",
    "useSealRecognition": "use_seal_recognition",
    "useOcrForImageBlock": "use_ocr_for_image_block",
    "layoutShapeMode": "layout_shape_mode",
    "promptLabel": "prompt_label",
    "repetitionPenalty": "repetition_penalty",
    "temperature": "temperature",
    "topP": "top_p",
    "minPixels": "min_pixels",
    "maxPixels": "max_pixels",
    "layoutNms": "layout_nms",
}

EXCLUDED_PARAMS = {"file", "fileType", "mergeTables", "relevelTitles", "restructurePages"}


def build_predict_kwargs(req: LayoutParsingRequest) -> dict:
    """Map AI Studio camelCase params to PaddleOCRVL snake_case kwargs."""
    req_dict = req.model_dump(exclude_none=True, exclude=EXCLUDED_PARAMS)
    return {
        pdx_name: req_dict[ai_name]
        for ai_name, pdx_name in PARAM_MAPPING.items()
        if ai_name in req_dict
    }


# ─── Image Utilities ────────────────────────────────────────────────

def image_to_base64_url(img_bytes: bytes, fmt: str = ".jpg") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


LAYOUT_COLORS = {
    "doc_title": (255, 0, 0), "paragraph_title": (0, 0, 255),
    "text": (0, 255, 0), "table": (255, 255, 0), "figure": (255, 0, 255),
    "figure_caption": (0, 255, 255), "image": (128, 0, 255),
    "image_box": (128, 0, 255), "chart": (0, 128, 255),
    "chart_box": (0, 128, 255), "formula": (255, 128, 0),
    "header": (128, 128, 128), "footer": (128, 128, 128),
    "aside_text": (128, 128, 128), "footnote": (128, 128, 128),
    "number": (128, 128, 128), "seal": (0, 200, 0),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
MARKITDOWN_EXTENSIONS = {
    ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt",
    ".html", ".htm", ".csv", ".json", ".xml", ".txt",
    ".rtf", ".epub", ".wav", ".mp3",
}


def draw_layout_visualization(img_array: np.ndarray, boxes: list) -> bytes:
    vis = img_array.copy()
    for box in boxes:
        coord = box.get("coordinate", [])
        label = box.get("label", "text")
        if len(coord) == 4:
            x1, y1, x2, y2 = [int(c) for c in coord]
            color = LAYOUT_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ─── Result Conversion ──────────────────────────────────────────────

def convert_result(res, page_idx: int, img_array: np.ndarray,
                   output_dir: str) -> dict:
    """Convert PaddleOCRVLResult to AI Studio layoutParsingResults format."""

    # 1. Structured JSON via save_to_json
    json_path = os.path.join(output_dir, f"page_{page_idx}.json")
    res.save_to_json(save_path=json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    # Build prunedResult (strip local-only keys)
    pruned_result = {k: v for k, v in res_json.items()
                     if k not in ("input_path", "page_index")}
    if "layout_det_res" in pruned_result and isinstance(pruned_result["layout_det_res"], dict):
        pruned_result["layout_det_res"] = {
            k: v for k, v in pruned_result["layout_det_res"].items()
            if k not in ("input_path", "page_index")
        }

    # 2. Markdown via save_to_markdown
    md_path = os.path.join(output_dir, f"page_{page_idx}.md")
    res.save_to_markdown(save_path=md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    markdown_images = {}
    imgs_dir = os.path.join(output_dir, "imgs")
    if os.path.isdir(imgs_dir):
        for img_name in os.listdir(imgs_dir):
            img_full = os.path.join(imgs_dir, img_name)
            with open(img_full, "rb") as f:
                markdown_images[f"imgs/{img_name}"] = image_to_base64_url(f.read())

    # 3. Output images (layout detection visualization)
    output_images = {}
    boxes = pruned_result.get("layout_det_res", {}).get("boxes", [])
    if boxes and img_array is not None:
        output_images["layout_det_res"] = image_to_base64_url(
            draw_layout_visualization(img_array, boxes)
        )

    # 4. Input image
    _, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    input_image = image_to_base64_url(buf.tobytes())

    return {
        "prunedResult": pruned_result,
        "markdown": {"text": markdown_text, "images": markdown_images},
        "outputImages": output_images,
        "inputImage": input_image,
    }


def process_with_markitdown(file_path: str, log_id: str = None) -> dict:
    """Process a file with markitdown. Returns response dict matching AI Studio format."""
    from markitdown import MarkItDown
    start_time = time.time()
    if log_id is None:
        log_id = str(uuid.uuid4())

    md_converter = MarkItDown()
    result = md_converter.convert(file_path)
    markdown_text = result.text_content

    elapsed = time.time() - start_time
    filename = os.path.basename(file_path)
    print(f"[{log_id}] markitdown: {filename} -> {len(markdown_text)} chars in {elapsed:.1f}s")

    return {
        "logId": log_id,
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": [{
                "prunedResult": {
                    "page_count": 1,
                    "width": 0,
                    "height": 0,
                    "model_settings": {"pipeline": "markitdown"},
                    "parsing_res_list": [],
                    "layout_det_res": {"boxes": []},
                },
                "markdown": {"text": markdown_text, "images": {}},
                "outputImages": {},
                "inputImage": "",
            }],
            "preprocessedImages": [],
            "dataInfo": {"type": "markitdown", "numPages": 1, "pages": [{"width": 0, "height": 0}]},
        },
    }


def load_pages_from_file(tmp_path: str, file_type: int, file_bytes: bytes,
                         dpi: int = 144) -> dict:
    """Load page images from PDF or image file. Returns {page_idx: np.ndarray}."""
    pages = {}
    if file_type == 0:
        import fitz
        doc = fitz.open(tmp_path)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            pages[i] = arr
        doc.close()
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            pages[0] = img
    return pages


def render_pages_to_temp_images(pages: dict, tmp_dir: str) -> dict:
    """Save rendered page images to temp PNG files. Returns {page_idx: path}."""
    paths = {}
    for i, arr in pages.items():
        path = os.path.join(tmp_dir, f"page_{i}.png")
        cv2.imwrite(path, arr)
        paths[i] = path
    return paths


def is_scanned_pdf(file_path: str, page_limit: int = 2) -> bool:
    """Check if a PDF is a scanned document (few text characters on first N pages)."""
    import fitz
    doc = fitz.open(file_path)
    try:
        total_chars = 0
        for i in range(min(page_limit, len(doc))):
            total_chars += len(doc[i].get_text().strip())
        return total_chars < SCANNED_PDF_CHAR_THRESHOLD
    finally:
        doc.close()


def get_file_processing_route(file_path: str) -> str:
    """Determine processing route: 'ocr', 'pdf_check', or 'markitdown'."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "ocr"
    if ext == ".pdf":
        return "pdf_check"
    return "markitdown"


# ─── Core Processing ────────────────────────────────────────────────

def process_single_file(req: LayoutParsingRequest, log_id: str | None = None) -> dict:
    """Process a single file. Supports base64 (file+fileType) or server path (filePath)."""
    start_time = time.time()
    if log_id is None:
        log_id = str(uuid.uuid4())

    # Resolve file source: filePath or base64
    use_path_directly = bool(req.filePath)
    if use_path_directly:
        file_path = req.filePath
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        # Auto-detect file type from extension
        ext = os.path.splitext(file_path)[1].lower()
        is_pdf = ext == ".pdf"
        file_type = 0 if is_pdf else 1
    else:
        file_bytes = base64.b64decode(req.file)
        file_type = req.fileType if req.fileType is not None else 1
        is_pdf = file_type == 0
        suffix = ".pdf" if is_pdf else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            file_path = tmp.name

    # Determine processing route
    route = get_file_processing_route(file_path)
    if route == "pdf_check":
        route = "ocr" if is_scanned_pdf(file_path) else "markitdown"

    # markitdown path: convert and return early
    if route == "markitdown":
        try:
            return process_with_markitdown(file_path, log_id)
        finally:
            if not use_path_directly:
                os.unlink(file_path)

    # OCR path: process through PaddleOCRVL
    tmp_path = file_path  # for cleanup in finally
    try:
        pipeline = get_pipeline()
        kwargs = build_predict_kwargs(req)

        original_pages = load_pages_from_file(tmp_path, file_type, file_bytes)

        if is_pdf and len(original_pages) > 0:
            img_tmp_dir = tempfile.mkdtemp(prefix="layout_pages_")
            try:
                page_paths = render_pages_to_temp_images(original_pages, img_tmp_dir)
                results = []
                for i, img_path in sorted(page_paths.items()):
                    page_results = list(pipeline.predict(img_path, **kwargs))
                    results.extend(page_results)
            finally:
                shutil.rmtree(img_tmp_dir, ignore_errors=True)
        else:
            results = list(pipeline.predict(tmp_path, **kwargs))

        if req.restructurePages:
            try:
                pipeline.restructure_pages(
                    results,
                    merge_tables=req.mergeTables,
                    relevel_titles=req.relevelTitles,
                )
            except Exception:
                pass

        all_layout_results = []
        for i, res in enumerate(results):
            page_img = original_pages.get(i)
            page_tmp_dir = tempfile.mkdtemp(prefix=f"layout_result_{i}_")
            try:
                all_layout_results.append(
                    convert_result(res, i, page_img, page_tmp_dir)
                )
            finally:
                shutil.rmtree(page_tmp_dir, ignore_errors=True)

        data_info = {"type": "pdf" if is_pdf else "image"}
        if is_pdf:
            data_info["numPages"] = len(all_layout_results)
            data_info["pages"] = []
            for r in all_layout_results:
                pr = r.get("prunedResult", {})
                data_info["pages"].append({
                    "width": pr.get("width", 0),
                    "height": pr.get("height", 0),
                })
        else:
            for a in original_pages.values():
                data_info["width"] = int(a.shape[1])
                data_info["height"] = int(a.shape[0])
                break

        elapsed = time.time() - start_time
        print(f"[{log_id}] {elapsed:.1f}s, {len(all_layout_results)} pages")

        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": all_layout_results,
                "preprocessedImages": [r["inputImage"] for r in all_layout_results],
                "dataInfo": data_info,
            },
        }
    finally:
        if not use_path_directly:
            os.unlink(tmp_path)


# ─── Batch Job Store ────────────────────────────────────────────────

class BatchJobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_job(self, total_files: int, callback_url: str | None) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "total_files": total_files,
                "completed_files": 0,
                "failed_files": 0,
                "results": [None] * total_files,
                "errors": [None] * total_files,
                "callback_url": callback_url,
                "created_at": time.time(),
            }
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def set_status(self, job_id: str, status: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status

    def set_file_result(self, job_id: str, file_index: int, result: dict):
        with self._lock:
            job = self._jobs[job_id]
            job["results"][file_index] = result
            job["completed_files"] += 1
            if job["completed_files"] + job["failed_files"] >= job["total_files"]:
                job["status"] = "completed"

    def set_file_error(self, job_id: str, file_index: int, error: str):
        with self._lock:
            job = self._jobs[job_id]
            job["errors"][file_index] = error
            job["failed_files"] += 1
            if job["completed_files"] + job["failed_files"] >= job["total_files"]:
                job["status"] = "completed" if job["completed_files"] > 0 else "failed"

    def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j["created_at"] < cutoff]
            for jid in expired:
                del self._jobs[jid]
        return len(expired)


# ─── Archive Helpers ─────────────────────────────────────────────

def _safe_extract_archive(archive_path: str, dest_dir: str) -> str:
    """Extract archive to dest_dir with path traversal protection."""
    os.makedirs(dest_dir, exist_ok=True)
    abs_dest = os.path.realpath(dest_dir)
    lower = archive_path.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(dest_dir, member.filename))
                if not member_path.startswith(abs_dest):
                    raise ValueError(f"Path traversal detected: {member.filename}")
            zf.extractall(dest_dir)
    elif lower.endswith((".tar.gz", ".tgz")):
        mode = "r:gz"
    elif lower.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif lower.endswith(".tar.xz"):
        mode = "r:xz"
    elif lower.endswith(".tar"):
        mode = "r:"
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    if not lower.endswith(".zip"):
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                member_path = os.path.realpath(os.path.join(dest_dir, member.name))
                if not member_path.startswith(abs_dest):
                    raise ValueError(f"Path traversal detected: {member.name}")
            tf.extractall(dest_dir)

    return dest_dir


def _compress_directory(source_dir: str, output_path: str) -> str:
    """Compress a directory into a zip file."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for f in files:
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, source_dir)
                zf.write(file_path, arcname)
    return output_path


def _count_archive_files(archive_path: str) -> int:
    """Count files in an archive without extracting."""
    lower = archive_path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            return sum(1 for info in zf.infolist() if not info.is_dir())
    elif lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else \
               "r:bz2" if lower.endswith(".tar.bz2") else \
               "r:xz" if lower.endswith(".tar.xz") else "r:"
        with tarfile.open(archive_path, mode) as tf:
            return sum(1 for m in tf.getmembers() if m.isfile())
    return 0


def _write_manifest(manifest_path: str, job_id: str):
    """Write the current manifest to disk."""
    job = archive_store.get_job(job_id)
    if not job:
        return
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(job["manifest"], f, ensure_ascii=False, indent=2)


# ─── Archive Job Store ───────────────────────────────────────────

class ArchiveJobStore:
    def __init__(self, jobs_dir: str):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._jobs_dir = jobs_dir

    def _save_job(self, job_id: str):
        """Persist a single job to disk (atomic write)."""
        job = self._jobs.get(job_id)
        if not job:
            return
        os.makedirs(self._jobs_dir, exist_ok=True)
        path = os.path.join(self._jobs_dir, f"{job_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
        os.replace(tmp, path)

    def _load_all(self):
        """Load all jobs from disk on startup."""
        if not os.path.isdir(self._jobs_dir):
            return
        count = 0
        for fname in os.listdir(self._jobs_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._jobs_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    job = json.load(f)
                jid = job["job_id"]
                self._jobs[jid] = job
                count += 1
            except Exception as e:
                print(f"[archive] Failed to load job from {path}: {e}")
        if count:
            print(f"[archive] Loaded {count} job(s) from disk")

    def _delete_job_file(self, job_id: str):
        """Remove a job's JSON file from disk."""
        path = os.path.join(self._jobs_dir, f"{job_id}.json")
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    def create_job(self, archive_name: str, extract_dir: str,
                   result_dir: str, total_files: int,
                   archive_path: str = "") -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "archive_name": archive_name,
                "archive_path": archive_path,
                "extract_dir": extract_dir,
                "result_dir": result_dir,
                "download_zip": None,
                "total_files": total_files,
                "completed_files": 0,
                "failed_files": 0,
                "manifest": [],
                "created_at": time.time(),
            }
            self._save_job(job_id)
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def set_status(self, job_id: str, status: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                self._save_job(job_id)

    def set_download_zip(self, job_id: str, zip_path: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["download_zip"] = zip_path
                self._save_job(job_id)

    def append_manifest_entry(self, job_id: str, entry: dict):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["manifest"].append(entry)
                if entry["status"] == "success":
                    job["completed_files"] += 1
                else:
                    job["failed_files"] += 1
                self._save_job(job_id)

    def get_incomplete_jobs(self) -> list[dict]:
        """Return jobs that were interrupted mid-processing."""
        with self._lock:
            return [copy.deepcopy(j) for j in self._jobs.values()
                    if j["status"] == "processing"]

    def cleanup_expired(self, max_age_seconds: float) -> list[str]:
        """Remove expired jobs, return download_zip paths for disk deletion."""
        cutoff = time.time() - max_age_seconds
        expired_zips = []
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j["created_at"] < cutoff]
            for jid in expired:
                zip_path = self._jobs[jid].get("download_zip")
                if zip_path:
                    expired_zips.append(zip_path)
                del self._jobs[jid]
                self._delete_job_file(jid)
        return expired_zips


# ─── Batch Worker & Callback ────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS)
batch_store = BatchJobStore()
archive_store = ArchiveJobStore(JOBS_DIR)


def _fire_callback(url: str, job: dict):
    try:
        payload = json.dumps({
            "jobId": job["job_id"],
            "status": job["status"],
            "totalFiles": job["total_files"],
            "completedFiles": job["completed_files"],
            "failedFiles": job["failed_files"],
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[batch] Callback failed for job {job['job_id']}: {e}")


def _process_batch_job(job_id: str, files: list[BatchFileRequest]):
    batch_store.set_status(job_id, "processing")
    for idx, file_req in enumerate(files):
        try:
            result = process_single_file(file_req)
            batch_store.set_file_result(job_id, idx, result)
        except Exception as e:
            traceback.print_exc()
            batch_store.set_file_error(job_id, idx, str(e))

    job = batch_store.get_job(job_id)
    if job and job.get("callback_url"):
        _fire_callback(job["callback_url"], job)


def _process_archive_job(job_id: str, archive_path: str):
    """Process an archive job: extract, process each file, compress results."""
    job = archive_store.get_job(job_id)
    if not job:
        return

    extract_dir = job["extract_dir"]
    result_dir = job["result_dir"]
    manifest_path = os.path.join(result_dir, "manifest.json")

    archive_store.set_status(job_id, "processing")

    try:
        # Extract archive
        _safe_extract_archive(archive_path, extract_dir)

        # Walk extracted files and process each
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, extract_dir)
                rel_no_ext, ext = os.path.splitext(rel_path)

                md_rel_path = rel_no_ext + ".md"
                json_rel_path = rel_no_ext + ".json"
                md_abs_path = os.path.join(result_dir, md_rel_path)
                json_abs_path = os.path.join(result_dir, json_rel_path)

                entry = {
                    "original_relative_path": rel_path,
                    "md_relative_path": md_rel_path,
                    "layout_relative_path": json_rel_path,
                    "status": "success",
                    "message": "",
                }

                try:
                    os.makedirs(os.path.dirname(md_abs_path), exist_ok=True)

                    req = LayoutParsingRequest(filePath=file_path)
                    result = process_single_file(req)

                    # Extract markdown text from result
                    md_text = ""
                    for page in result.get("result", {}).get("layoutParsingResults", []):
                        page_md = page.get("markdown", {}).get("text", "")
                        if page_md:
                            md_text += page_md + "\n\n"

                    # Save markdown
                    with open(md_abs_path, "w", encoding="utf-8") as f:
                        f.write(md_text)

                    # Save layout JSON only for OCR results
                    data_type = result.get("result", {}).get("dataInfo", {}).get("type", "")
                    if data_type != "markitdown":
                        with open(json_abs_path, "w", encoding="utf-8") as f:
                            json.dump(
                                result["result"]["layoutParsingResults"],
                                f, ensure_ascii=False, indent=2,
                            )
                    else:
                        entry["layout_relative_path"] = ""

                except Exception as e:
                    traceback.print_exc()
                    entry["status"] = "failed"
                    entry["message"] = str(e)
                    entry["md_relative_path"] = ""
                    entry["layout_relative_path"] = ""

                # Update manifest after each file
                archive_store.append_manifest_entry(job_id, entry)
                _write_manifest(manifest_path, job_id)

        # Compress result directory
        archive_name = job["archive_name"]
        download_zip = os.path.join(
            UPLOAD_DIR, f"{archive_name}_{int(time.time())}_result.zip"
        )
        _compress_directory(result_dir, download_zip)
        archive_store.set_download_zip(job_id, download_zip)

        # Cleanup: delete extract dir, result dir, original archive
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        try:
            os.unlink(archive_path)
        except OSError:
            pass

        archive_store.set_status(job_id, "completed")

    except Exception as e:
        traceback.print_exc()
        archive_store.set_status(job_id, "failed")


def _resume_archive_job(job_id: str):
    """Resume an interrupted archive job from where it left off."""
    job = archive_store.get_job(job_id)
    if not job:
        return

    archive_path = job.get("archive_path", "")
    extract_dir = job["extract_dir"]
    result_dir = job["result_dir"]

    # Edge case: compression was done but status not set to completed
    existing_zip = job.get("download_zip")
    if existing_zip and os.path.exists(existing_zip):
        archive_store.set_status(job_id, "completed")
        print(f"[archive] Job {job_id} was already finished, marked completed")
        return
    manifest_path = os.path.join(result_dir, "manifest.json")

    # Collect already-processed files from persisted manifest
    processed = {e["original_relative_path"] for e in job["manifest"]}
    remaining = job["total_files"] - job["completed_files"] - job["failed_files"]

    print(f"[archive] Resuming job {job_id}: {len(processed)} done, {remaining} remaining")

    if remaining <= 0:
        # All files were processed but job didn't finish (e.g. crashed during compression)
        # Just re-compress and complete
        pass
    elif not archive_path or not os.path.exists(archive_path):
        print(f"[archive] Cannot resume job {job_id}: archive file missing ({archive_path})")
        archive_store.set_status(job_id, "failed")
        return
    else:
        # Re-extract if extract_dir was cleaned up
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            _safe_extract_archive(archive_path, extract_dir)

        os.makedirs(result_dir, exist_ok=True)

        # Process remaining files
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, extract_dir)

                if rel_path in processed:
                    continue

                rel_no_ext, ext = os.path.splitext(rel_path)
                md_rel_path = rel_no_ext + ".md"
                json_rel_path = rel_no_ext + ".json"
                md_abs_path = os.path.join(result_dir, md_rel_path)
                json_abs_path = os.path.join(result_dir, json_rel_path)

                entry = {
                    "original_relative_path": rel_path,
                    "md_relative_path": md_rel_path,
                    "layout_relative_path": json_rel_path,
                    "status": "success",
                    "message": "",
                }

                try:
                    os.makedirs(os.path.dirname(md_abs_path), exist_ok=True)
                    req = LayoutParsingRequest(filePath=file_path)
                    result = process_single_file(req)

                    md_text = ""
                    for page in result.get("result", {}).get("layoutParsingResults", []):
                        page_md = page.get("markdown", {}).get("text", "")
                        if page_md:
                            md_text += page_md + "\n\n"

                    with open(md_abs_path, "w", encoding="utf-8") as f:
                        f.write(md_text)

                    data_type = result.get("result", {}).get("dataInfo", {}).get("type", "")
                    if data_type != "markitdown":
                        with open(json_abs_path, "w", encoding="utf-8") as f:
                            json.dump(
                                result["result"]["layoutParsingResults"],
                                f, ensure_ascii=False, indent=2,
                            )
                    else:
                        entry["layout_relative_path"] = ""

                except Exception as e:
                    traceback.print_exc()
                    entry["status"] = "failed"
                    entry["message"] = str(e)
                    entry["md_relative_path"] = ""
                    entry["layout_relative_path"] = ""

                archive_store.append_manifest_entry(job_id, entry)
                _write_manifest(manifest_path, job_id)

    # Compress and finalize (same as _process_archive_job)
    try:
        archive_name = job["archive_name"]
        download_zip = os.path.join(
            UPLOAD_DIR, f"{archive_name}_{int(time.time())}_result.zip"
        )
        _compress_directory(result_dir, download_zip)
        archive_store.set_download_zip(job_id, download_zip)

        # Cleanup intermediates
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        if archive_path and os.path.exists(archive_path):
            try:
                os.unlink(archive_path)
            except OSError:
                pass

        archive_store.set_status(job_id, "completed")
        print(f"[archive] Job {job_id} resumed and completed")

    except Exception as e:
        traceback.print_exc()
        archive_store.set_status(job_id, "failed")
        print(f"[archive] Job {job_id} resume failed: {e}")


# ─── Cleanup ────────────────────────────────────────────────────────

_cleanup_stop = threading.Event()


def _cleanup_loop():
    while not _cleanup_stop.wait(timeout=300):
        removed = batch_store.cleanup_expired(JOB_TTL_SECONDS)
        if removed:
            print(f"[batch] Cleaned up {removed} expired jobs")
        expired_zips = archive_store.cleanup_expired(ARCHIVE_RESULT_TTL_SECONDS)
        for zip_path in expired_zips:
            try:
                if os.path.exists(zip_path):
                    os.unlink(zip_path)
                    print(f"[archive] Deleted expired result: {zip_path}")
            except OSError as e:
                print(f"[archive] Failed to delete {zip_path}: {e}")


# ─── FastAPI App ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)
    _ensure_admin()

    # Load persisted jobs and resume interrupted ones
    archive_store._load_all()
    for job in archive_store.get_incomplete_jobs():
        print(f"[archive] Resuming interrupted job {job['job_id']}")
        executor.submit(_resume_archive_job, job["job_id"])

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()
    yield
    _cleanup_stop.set()
    executor.shutdown(wait=False)


app = FastAPI(
    title="PaddleOCR-VL Layout Parsing Server",
    lifespan=lifespan,
    dependencies=[Depends(verify_token)],
)


@app.post("/layout-parsing")
async def layout_parsing(req: LayoutParsingRequest):
    log_id = str(uuid.uuid4())
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, process_single_file, req, log_id
        )
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "logId": log_id, "errorCode": -1, "errorMsg": str(e), "result": None,
        })


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Batch Endpoints ────────────────────────────────────────────────

@app.post("/batch/layout-parsing", status_code=202)
async def batch_submit(req: BatchSubmitRequest):
    job_id = batch_store.create_job(
        total_files=len(req.files),
        callback_url=req.callbackUrl,
    )
    executor.submit(_process_batch_job, job_id, req.files)
    return {
        "jobId": job_id,
        "status": "pending",
        "totalFiles": len(req.files),
    }


@app.get("/batch/status/{job_id}")
async def batch_status(job_id: str):
    job = batch_store.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    total = job["total_files"]
    done = job["completed_files"] + job["failed_files"]
    return {
        "jobId": job["job_id"],
        "status": job["status"],
        "totalFiles": job["total_files"],
        "completedFiles": job["completed_files"],
        "failedFiles": job["failed_files"],
        "progress": round(done / total, 2) if total > 0 else 0.0,
    }


@app.get("/batch/result/{job_id}")
async def batch_result(job_id: str):
    job = batch_store.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    if job["status"] not in ("completed", "failed"):
        total = job["total_files"]
        done = job["completed_files"] + job["failed_files"]
        return JSONResponse(status_code=409, content={
            "error": "Job not yet completed",
            "status": job["status"],
            "progress": round(done / total, 2) if total > 0 else 0.0,
        })

    results = []
    for idx in range(job["total_files"]):
        file_result = {
            "fileIndex": idx,
            "status": "success" if job["results"][idx] is not None else "failed",
        }
        if job["results"][idx] is not None:
            file_result["result"] = job["results"][idx]
        else:
            file_result["error"] = job["errors"][idx]
        results.append(file_result)

    return {
        "jobId": job["job_id"],
        "status": job["status"],
        "totalFiles": job["total_files"],
        "completedFiles": job["completed_files"],
        "failedFiles": job["failed_files"],
        "results": results,
    }


# ─── Archive Endpoints ───────────────────────────────────────────

@app.post("/batch/archive-parsing", status_code=202)
async def archive_parsing(file: UploadFile = FastAPIFile(...)):
    filename = file.filename or "archive.zip"
    lower_name = filename.lower()
    supported = (".zip", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar")
    if not any(lower_name.endswith(ext) for ext in supported):
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported archive format. Use: {', '.join(supported)}",
        })

    content = await file.read()

    # Strip compound extensions like .tar.gz
    base_name = filename
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
        if base_name.lower().endswith(ext):
            base_name = base_name[:len(base_name) - len(ext)]
            break

    timestamp = int(time.time())
    work_dir_name = f"{base_name}_{timestamp}"
    extract_dir = os.path.join(UPLOAD_DIR, work_dir_name)
    result_dir = os.path.join(UPLOAD_DIR, f"{work_dir_name}_result")

    # Save uploaded archive
    archive_ext = filename[len(base_name):]  # preserves original extension
    archive_path = os.path.join(UPLOAD_DIR, f"{work_dir_name}{archive_ext}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(archive_path, "wb") as f:
        f.write(content)

    total_files = _count_archive_files(archive_path)

    job_id = archive_store.create_job(
        archive_name=base_name,
        extract_dir=extract_dir,
        result_dir=result_dir,
        total_files=total_files,
        archive_path=archive_path,
    )
    executor.submit(_process_archive_job, job_id, archive_path)

    return {
        "jobId": job_id,
        "status": "pending",
        "totalFiles": total_files,
    }


@app.get("/batch/archive-jobs")
async def archive_list_jobs():
    jobs = []
    with archive_store._lock:
        for j in archive_store._jobs.values():
            total = j["total_files"]
            done = j["completed_files"] + j["failed_files"]
            jobs.append({
                "jobId": j["job_id"],
                "archiveName": j["archive_name"],
                "status": j["status"],
                "totalFiles": total,
                "completedFiles": j["completed_files"],
                "failedFiles": j["failed_files"],
                "progress": round(done / total, 2) if total > 0 else 0.0,
                "createdAt": j["created_at"],
            })
    return {"jobs": jobs}


@app.get("/batch/archive-status/{job_id}")
async def archive_status(job_id: str):
    job = archive_store.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    total = job["total_files"]
    done = job["completed_files"] + job["failed_files"]
    return {
        "jobId": job["job_id"],
        "status": job["status"],
        "totalFiles": total,
        "completedFiles": job["completed_files"],
        "failedFiles": job["failed_files"],
        "progress": round(done / total, 2) if total > 0 else 0.0,
    }


@app.get("/batch/archive-download/{job_id}")
async def archive_download(job_id: str):
    job = archive_store.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    if job["status"] != "completed":
        return JSONResponse(status_code=409, content={
            "error": "Job not yet completed",
            "status": job["status"],
        })

    download_zip = job.get("download_zip")
    if not download_zip or not os.path.exists(download_zip):
        return JSONResponse(status_code=410, content={
            "error": "Result file no longer available",
        })

    return FileResponse(
        path=download_zip,
        media_type="application/zip",
        filename=f"{job['archive_name']}_result.zip",
    )


# ─── Auth Admin Endpoints ────────────────────────────────────────

class TokenRequest(BaseModel):
    email: str
    expires_in: Optional[int] = None  # seconds, None = never expires


@app.post("/auth/token")
async def auth_create_token(req: TokenRequest, request: Request):
    admin_err = _require_admin(request)
    if admin_err:
        return admin_err
    try:
        # Create user if not exists
        users = _load_users()
        if req.email not in users:
            users[req.email] = {
                "salt": secrets.token_hex(16),
                "role": "user",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_users(users)
        token, expires_at = create_token(req.email, req.expires_in)
        return {"email": req.email, "token": token, "expires_at": expires_at}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/auth/token/refresh")
async def auth_refresh_token(req: TokenRequest, request: Request):
    admin_err = _require_admin(request)
    if admin_err:
        return admin_err
    users = _load_users()
    if req.email not in users:
        return JSONResponse(status_code=404, content={"error": "User not found"})
    # Generate new salt to revoke old tokens
    users[req.email]["salt"] = secrets.token_hex(16)
    _save_users(users)
    token, expires_at = create_token(req.email, req.expires_in)
    return {"email": req.email, "token": token, "expires_at": expires_at}


@app.delete("/auth/token/{email}")
async def auth_delete_user(email: str, request: Request):
    admin_err = _require_admin(request)
    if admin_err:
        return admin_err
    users = _load_users()
    if email not in users:
        return JSONResponse(status_code=404, content={"error": "User not found"})
    if email == ADMIN_EMAIL:
        return JSONResponse(status_code=400, content={"error": "Cannot delete admin user"})
    del users[email]
    _save_users(users)
    return {"email": email, "status": "deleted"}


@app.get("/auth/users")
async def auth_list_users(request: Request):
    admin_err = _require_admin(request)
    if admin_err:
        return admin_err
    users = _load_users()
    return [
        {"email": email, "role": info["role"], "created_at": info["created_at"]}
        for email, info in users.items()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
