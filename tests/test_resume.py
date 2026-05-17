import json
import os
import zipfile
from unittest.mock import patch, MagicMock

import pytest

import server
from server import ArchiveJobStore, _resume_archive_job, _process_archive_job


def _make_ocr_result(md_text="extracted text"):
    """Build a minimal dict that matches what process_single_file returns."""
    return {
        "result": {
            "dataInfo": {"type": "ocr"},
            "layoutParsingResults": [
                {
                    "markdown": {"text": md_text},
                    "prunedResult": {
                        "layout_det_res": {
                            "boxes": [{"score": 0.95}]
                        }
                    },
                }
            ],
        }
    }


@pytest.fixture
def archive_env(tmp_path):
    """Set up a complete archive processing environment with mocked OCR."""
    uploads = str(tmp_path / "uploads")
    jobs_dir = str(tmp_path / "uploads" / "jobs")
    os.makedirs(uploads, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)

    # Create a zip with 3 files
    zip_path = os.path.join(uploads, "test.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "aaa")
        zf.writestr("b.txt", "bbb")
        zf.writestr("c.txt", "ccc")

    # Monkey-patch the global store and UPLOAD_DIR
    store = ArchiveJobStore(jobs_dir)
    old_store = server.archive_store
    old_upload = server.UPLOAD_DIR
    server.archive_store = store
    server.UPLOAD_DIR = uploads

    yield {
        "store": store,
        "uploads": uploads,
        "jobs_dir": jobs_dir,
        "zip_path": zip_path,
    }

    server.archive_store = old_store
    server.UPLOAD_DIR = old_upload


class TestProcessArchiveJob:
    @patch("server.process_single_file", return_value=_make_ocr_result("hello"))
    def test_full_processing(self, mock_ocr, archive_env):
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]

        jid = store.create_job(
            "test", os.path.join(uploads, "ext"), os.path.join(uploads, "res"),
            3, archive_path=zip_path,
        )

        _process_archive_job(jid, zip_path)

        job = store.get_job(jid)
        assert job["status"] == "completed"
        assert job["completed_files"] == 3
        assert job["failed_files"] == 0
        assert len(job["manifest"]) == 3
        assert job["download_zip"] is not None
        assert os.path.exists(job["download_zip"])

        # OCR was called 3 times
        assert mock_ocr.call_count == 3

    @patch("server.process_single_file", side_effect=Exception("OCR boom"))
    def test_all_files_fail(self, mock_ocr, archive_env):
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]

        jid = store.create_job(
            "test", os.path.join(uploads, "ext"), os.path.join(uploads, "res"),
            3, archive_path=zip_path,
        )

        _process_archive_job(jid, zip_path)

        job = store.get_job(jid)
        assert job["status"] == "completed"  # job still completes, files just fail
        assert job["completed_files"] == 0
        assert job["failed_files"] == 3


class TestResumeArchiveJob:
    @patch("server.process_single_file", return_value=_make_ocr_result("resumed"))
    def test_resume_skips_processed_files(self, mock_ocr, archive_env):
        """Resume should skip files already in the manifest."""
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]

        extract_dir = os.path.join(uploads, "ext")
        result_dir = os.path.join(uploads, "res")

        jid = store.create_job("test", extract_dir, result_dir, 3, archive_path=zip_path)

        # Simulate partial processing: 1 of 3 files done
        store.set_status(jid, "processing")
        store.append_manifest_entry(jid, {
            "original_relative_path": "a.txt",
            "md_relative_path": "a.md",
            "layout_relative_path": "a.json",
            "status": "success",
            "message": "",
        })

        # Resume
        _resume_archive_job(jid)

        job = store.get_job(jid)
        assert job["status"] == "completed"
        assert job["completed_files"] == 3  # 1 already done + 2 new (b.txt, c.txt)
        assert job["failed_files"] == 0
        assert len(job["manifest"]) == 3
        # OCR called for b.txt and c.txt only, not a.txt
        assert mock_ocr.call_count == 2

    @patch("server.process_single_file", return_value=_make_ocr_result("x"))
    def test_resume_when_extract_dir_missing(self, mock_ocr, archive_env):
        """Resume should re-extract if extract_dir was cleaned up."""
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]

        extract_dir = os.path.join(uploads, "ext")
        result_dir = os.path.join(uploads, "res")

        jid = store.create_job("test", extract_dir, result_dir, 3, archive_path=zip_path)
        store.set_status(jid, "processing")
        store.append_manifest_entry(jid, {
            "original_relative_path": "a.txt",
            "md_relative_path": "a.md",
            "layout_relative_path": "a.json",
            "status": "success",
            "message": "",
        })

        # Delete extract_dir to simulate cleanup
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)

        _resume_archive_job(jid)

        job = store.get_job(jid)
        assert job["status"] == "completed"
        assert mock_ocr.call_count == 2

    @patch("server.process_single_file", return_value=_make_ocr_result("x"))
    def test_resume_all_files_done_crashed_during_compress(self, mock_ocr, archive_env):
        """All files processed but crashed before compression — resume should just compress."""
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]

        extract_dir = os.path.join(uploads, "ext")
        result_dir = os.path.join(uploads, "res")

        jid = store.create_job("test", extract_dir, result_dir, 3, archive_path=zip_path)
        store.set_status(jid, "processing")
        # All 3 files in manifest
        for name in ["a.txt", "b.txt", "c.txt"]:
            store.append_manifest_entry(jid, {
                "original_relative_path": name,
                "md_relative_path": name.replace(".txt", ".md"),
                "layout_relative_path": name.replace(".txt", ".json"),
                "status": "success",
                "message": "",
            })

        # Ensure result_dir has some content to compress
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "a.md"), "w") as f:
            f.write("aaa")

        _resume_archive_job(jid)

        job = store.get_job(jid)
        assert job["status"] == "completed"
        assert job["download_zip"] is not None
        assert os.path.exists(job["download_zip"])
        mock_ocr.assert_not_called()  # no files to process

    def test_resume_download_zip_already_exists(self, archive_env):
        """Edge case: zip exists but status not set — just mark completed."""
        store = archive_env["store"]
        uploads = archive_env["uploads"]

        jid = store.create_job("test", "/e", "/r", 3, archive_path="/a.zip")
        store.set_status(jid, "processing")

        # Create a fake result zip
        zip_path = os.path.join(uploads, "already_done.zip")
        with open(zip_path, "w") as f:
            f.write("fake")
        store.set_download_zip(jid, zip_path)

        _resume_archive_job(jid)

        job = store.get_job(jid)
        assert job["status"] == "completed"

    def test_resume_archive_file_missing(self, archive_env):
        """Archive file was deleted — mark as failed."""
        store = archive_env["store"]

        jid = store.create_job("test", "/e", "/r", 3, archive_path="/nonexistent.zip")
        store.set_status(jid, "processing")

        _resume_archive_job(jid)

        job = store.get_job(jid)
        assert job["status"] == "failed"

    def test_resume_nonexistent_job(self, archive_env):
        """Resume a job that doesn't exist — should be a no-op."""
        _resume_archive_job("no-such-id")  # should not raise


class TestPersistenceAcrossRestart:
    @patch("server.process_single_file", return_value=_make_ocr_result("persisted"))
    def test_survive_store_restart(self, mock_ocr, archive_env):
        """Simulate: process 1 file -> 'restart' -> resume remaining."""
        store = archive_env["store"]
        zip_path = archive_env["zip_path"]
        uploads = archive_env["uploads"]
        jobs_dir = archive_env["jobs_dir"]

        extract_dir = os.path.join(uploads, "ext")
        result_dir = os.path.join(uploads, "res")

        jid = store.create_job("test", extract_dir, result_dir, 3, archive_path=zip_path)
        store.set_status(jid, "processing")
        store.append_manifest_entry(jid, {
            "original_relative_path": "a.txt",
            "md_relative_path": "a.md",
            "layout_relative_path": "a.json",
            "status": "success",
            "message": "",
        })

        # Simulate restart: create new store, load from disk
        store2 = ArchiveJobStore(jobs_dir)
        store2._load_all()

        job = store2.get_job(jid)
        assert job is not None
        assert job["status"] == "processing"
        assert len(job["manifest"]) == 1

        # Patch global store for resume
        server.archive_store = store2
        try:
            _resume_archive_job(jid)
        finally:
            server.archive_store = archive_env["store"]

        job = store2.get_job(jid)
        assert job["status"] == "completed"
        assert job["completed_files"] == 3  # 1 from manifest + 2 newly processed
        assert mock_ocr.call_count == 2
