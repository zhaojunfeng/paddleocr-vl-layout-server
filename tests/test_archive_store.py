import json
import os
import time

import pytest

from server import ArchiveJobStore


class TestCreateJob:
    def test_returns_uuid(self, store):
        jid = store.create_job("test", "/ext", "/res", 5)
        assert len(jid) == 36  # UUID format
        assert "-" in jid

    def test_initial_status_pending(self, store):
        jid = store.create_job("test", "/ext", "/res", 5)
        job = store.get_job(jid)
        assert job["status"] == "pending"
        assert job["total_files"] == 5
        assert job["completed_files"] == 0
        assert job["failed_files"] == 0
        assert job["manifest"] == []
        assert job["download_zip"] is None

    def test_stores_archive_path(self, store):
        jid = store.create_job("test", "/ext", "/res", 3, archive_path="/tmp/a.zip")
        job = store.get_job(jid)
        assert job["archive_path"] == "/tmp/a.zip"

    def test_persists_to_disk(self, store, jobs_dir):
        jid = store.create_job("test", "/ext", "/res", 2)
        json_path = os.path.join(jobs_dir, f"{jid}.json")
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["job_id"] == jid
        assert data["status"] == "pending"


class TestGetJob:
    def test_returns_none_for_missing(self, store):
        assert store.get_job("nonexistent") is None

    def test_returns_deep_copy(self, store):
        jid = store.create_job("test", "/ext", "/res", 1)
        job1 = store.get_job(jid)
        job1["status"] = "mutated"
        job2 = store.get_job(jid)
        assert job2["status"] == "pending"


class TestSetStatus:
    def test_updates_status(self, store):
        jid = store.create_job("test", "/ext", "/res", 1)
        store.set_status(jid, "processing")
        assert store.get_job(jid)["status"] == "processing"

    def test_persists_status_change(self, store, jobs_dir):
        jid = store.create_job("test", "/ext", "/res", 1)
        store.set_status(jid, "completed")
        with open(os.path.join(jobs_dir, f"{jid}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "completed"

    def test_noop_for_missing_job(self, store):
        store.set_status("nonexistent", "failed")  # should not raise


class TestSetDownloadZip:
    def test_sets_path(self, store):
        jid = store.create_job("test", "/ext", "/res", 1)
        store.set_download_zip(jid, "/tmp/result.zip")
        assert store.get_job(jid)["download_zip"] == "/tmp/result.zip"

    def test_persists(self, store, jobs_dir):
        jid = store.create_job("test", "/ext", "/res", 1)
        store.set_download_zip(jid, "/tmp/result.zip")
        with open(os.path.join(jobs_dir, f"{jid}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["download_zip"] == "/tmp/result.zip"


class TestAppendManifestEntry:
    def test_appends_success_entry(self, store):
        jid = store.create_job("test", "/ext", "/res", 3)
        store.append_manifest_entry(jid, {
            "original_relative_path": "a.txt",
            "md_relative_path": "a.md",
            "layout_relative_path": "a.json",
            "status": "success",
            "message": "",
        })
        job = store.get_job(jid)
        assert len(job["manifest"]) == 1
        assert job["completed_files"] == 1
        assert job["failed_files"] == 0

    def test_appends_failed_entry(self, store):
        jid = store.create_job("test", "/ext", "/res", 3)
        store.append_manifest_entry(jid, {
            "original_relative_path": "b.txt",
            "md_relative_path": "",
            "layout_relative_path": "",
            "status": "failed",
            "message": "OCR error",
        })
        job = store.get_job(jid)
        assert job["completed_files"] == 0
        assert job["failed_files"] == 1

    def test_mixed_entries(self, store):
        jid = store.create_job("test", "/ext", "/res", 3)
        store.append_manifest_entry(jid, {"original_relative_path": "a", "status": "success", "md_relative_path": "", "layout_relative_path": "", "message": ""})
        store.append_manifest_entry(jid, {"original_relative_path": "b", "status": "failed", "md_relative_path": "", "layout_relative_path": "", "message": "err"})
        store.append_manifest_entry(jid, {"original_relative_path": "c", "status": "success", "md_relative_path": "", "layout_relative_path": "", "message": ""})
        job = store.get_job(jid)
        assert job["completed_files"] == 2
        assert job["failed_files"] == 1
        assert len(job["manifest"]) == 3

    def test_persists_manifest(self, store, jobs_dir):
        jid = store.create_job("test", "/ext", "/res", 1)
        store.append_manifest_entry(jid, {"original_relative_path": "a", "status": "success", "md_relative_path": "", "layout_relative_path": "", "message": ""})
        with open(os.path.join(jobs_dir, f"{jid}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["manifest"]) == 1


class TestGetIncompleteJobs:
    def test_returns_processing_jobs(self, store):
        jid1 = store.create_job("a", "/e", "/r", 1)
        jid2 = store.create_job("b", "/e", "/r", 1)
        jid3 = store.create_job("c", "/e", "/r", 1)
        store.set_status(jid1, "processing")
        store.set_status(jid2, "completed")
        # jid3 stays "pending"

        incomplete = store.get_incomplete_jobs()
        ids = {j["job_id"] for j in incomplete}
        assert ids == {jid1}

    def test_empty_when_all_done(self, store):
        jid = store.create_job("a", "/e", "/r", 1)
        store.set_status(jid, "completed")
        assert store.get_incomplete_jobs() == []


class TestCleanupExpired:
    def test_removes_old_jobs(self, store):
        jid = store.create_job("a", "/e", "/r", 1)
        # Backdate created_at
        store._jobs[jid]["created_at"] = time.time() - 999999
        expired_zips = store.cleanup_expired(3600)
        assert store.get_job(jid) is None
        assert expired_zips == []

    def test_returns_download_zip_paths(self, store, jobs_dir):
        jid = store.create_job("a", "/e", "/r", 1)
        store.set_download_zip(jid, "/tmp/result.zip")
        store._jobs[jid]["created_at"] = time.time() - 999999
        expired_zips = store.cleanup_expired(3600)
        assert "/tmp/result.zip" in expired_zips

    def test_deletes_json_file(self, store, jobs_dir):
        jid = store.create_job("a", "/e", "/r", 1)
        json_path = os.path.join(jobs_dir, f"{jid}.json")
        assert os.path.exists(json_path)
        store._jobs[jid]["created_at"] = time.time() - 999999
        store.cleanup_expired(3600)
        assert not os.path.exists(json_path)

    def test_keeps_recent_jobs(self, store):
        jid = store.create_job("a", "/e", "/r", 1)
        expired_zips = store.cleanup_expired(3600)
        assert store.get_job(jid) is not None
        assert expired_zips == []


class TestLoadAll:
    def test_loads_persisted_jobs(self, store, jobs_dir):
        # Create jobs and let them persist
        jid1 = store.create_job("a", "/e", "/r", 2)
        jid2 = store.create_job("b", "/e", "/r", 3)
        store.set_status(jid1, "processing")

        # Create a new store from the same directory
        store2 = ArchiveJobStore(jobs_dir)
        store2._load_all()

        assert store2.get_job(jid1) is not None
        assert store2.get_job(jid2) is not None
        assert store2.get_job(jid1)["status"] == "processing"
        assert store2.get_job(jid2)["status"] == "pending"

    def test_skips_non_json_files(self, store, jobs_dir):
        # Create a non-json file in the jobs dir
        with open(os.path.join(jobs_dir, "README.txt"), "w") as f:
            f.write("not a job")

        jid = store.create_job("a", "/e", "/r", 1)

        store2 = ArchiveJobStore(jobs_dir)
        store2._load_all()
        assert store2.get_job(jid) is not None

    def test_handles_corrupt_json(self, store, jobs_dir):
        # Write corrupt JSON
        with open(os.path.join(jobs_dir, "bad.json"), "w") as f:
            f.write("{invalid json")

        jid = store.create_job("a", "/e", "/r", 1)

        store2 = ArchiveJobStore(jobs_dir)
        store2._load_all()
        assert store2.get_job(jid) is not None  # valid job still loaded

    def test_empty_dir(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        store = ArchiveJobStore(empty)
        store._load_all()  # should not raise
        assert store.get_job("anything") is None
