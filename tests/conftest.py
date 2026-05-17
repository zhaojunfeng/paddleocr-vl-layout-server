import os
import sys
import json
import zipfile
import types
from unittest.mock import MagicMock

import pytest

# ─── Mock heavy dependencies before importing server ───
_MOCK_MODULES = [
    "cv2", "numpy", "paddle", "paddleocr", "paddlepaddle",
    "fitz", "markitdown", "doc_parser",
    "paddleocr.paddleocr", "paddleocr.tools",
    "markitdown._markitdown",
]
for mod in _MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Ensure server.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server import ArchiveJobStore  # noqa: E402


@pytest.fixture
def jobs_dir(tmp_path):
    """Provide a temporary jobs directory."""
    d = tmp_path / "jobs"
    d.mkdir()
    return str(d)


@pytest.fixture
def store(jobs_dir):
    """Provide a fresh ArchiveJobStore backed by a temp directory."""
    return ArchiveJobStore(jobs_dir)


@pytest.fixture
def upload_dir(tmp_path):
    """Provide a temporary upload directory."""
    d = tmp_path / "uploads"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_zip(upload_dir):
    """Create a minimal zip archive with a few text files, return its path."""
    zip_path = os.path.join(upload_dir, "test_archive.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", "content of file 1")
        zf.writestr("subdir/file2.txt", "content of file 2")
        zf.writestr("subdir/file3.txt", "content of file 3")
    return zip_path
