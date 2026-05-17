import io
import json
import os
import zipfile
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import server


def _make_ocr_result(md_text="test text"):
    return {
        "result": {
            "dataInfo": {"type": "ocr"},
            "layoutParsingResults": [
                {
                    "markdown": {"text": md_text},
                    "prunedResult": {
                        "layout_det_res": {"boxes": [{"score": 0.9}]}
                    },
                }
            ],
        }
    }


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Isolate each test with its own upload dir, jobs dir, and user db."""
    uploads = str(tmp_path / "uploads")
    jobs_dir = str(tmp_path / "uploads" / "jobs")
    user_db = str(tmp_path / "users.json")
    os.makedirs(uploads, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)

    monkeypatch.setattr(server, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(server, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(server, "USER_DB_PATH", user_db)

    # Replace global store with isolated one
    store = server.ArchiveJobStore(jobs_dir)
    monkeypatch.setattr(server, "archive_store", store)

    # Ensure admin user exists
    server._ensure_admin()

    return {
        "uploads": uploads,
        "jobs_dir": jobs_dir,
        "store": store,
    }


@pytest.fixture
def auth_header():
    """Generate a valid JWT token for the admin user."""
    token, _ = server.create_token(server.ADMIN_EMAIL)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    """Create a TestClient (auth dependency is on the app, not per-request)."""
    return TestClient(server.app, raise_server_exceptions=False)


class TestHealth:
    def test_health(self, client, auth_header):
        res = client.get("/health", headers=auth_header)
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


class TestArchiveJobs:
    def test_list_empty(self, client, auth_header):
        res = client.get("/batch/archive-jobs", headers=auth_header)
        assert res.status_code == 200
        assert res.json()["jobs"] == []

    def test_list_after_create(self, client, auth_header, isolated_env):
        store = isolated_env["store"]
        store.create_job("test.zip", "/ext", "/res", 5)

        res = client.get("/batch/archive-jobs", headers=auth_header)
        assert res.status_code == 200
        jobs = res.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["archiveName"] == "test.zip"
        assert jobs[0]["totalFiles"] == 5


class TestArchiveStatus:
    def test_status_not_found(self, client, auth_header):
        res = client.get("/batch/archive-status/nonexistent", headers=auth_header)
        assert res.status_code == 404

    def test_status_pending(self, client, auth_header, isolated_env):
        store = isolated_env["store"]
        jid = store.create_job("test.zip", "/ext", "/res", 3)

        res = client.get(f"/batch/archive-status/{jid}", headers=auth_header)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "pending"
        assert data["totalFiles"] == 3
        assert data["completedFiles"] == 0
        assert data["progress"] == 0.0


class TestArchiveParsing:
    def test_unsupported_format(self, client, auth_header):
        res = client.post(
            "/batch/archive-parsing",
            headers=auth_header,
            files={"file": ("test.rar", b"fake", "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "Unsupported" in res.json()["error"]

    @patch("server._process_archive_job")
    def test_submit_zip(self, mock_process, client, auth_header, isolated_env):
        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, "w") as zf:
            zf.writestr("a.txt", "hello")
        zip_bytes = zip_content.getvalue()

        res = client.post(
            "/batch/archive-parsing",
            headers=auth_header,
            files={"file": ("test.zip", zip_bytes, "application/zip")},
        )
        assert res.status_code == 202
        data = res.json()
        assert "jobId" in data
        assert data["status"] == "pending"
        assert data["totalFiles"] == 1

        # Job should be persisted to disk
        store = isolated_env["store"]
        job = store.get_job(data["jobId"])
        assert job is not None
        assert job["archive_path"].endswith(".zip")

    @patch("server._process_archive_job")
    def test_submit_tar_gz(self, mock_process, client, auth_header, isolated_env):
        import tarfile
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tf:
            info = tarfile.TarInfo(name="a.txt")
            data = b"hello"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        res = client.post(
            "/batch/archive-parsing",
            headers=auth_header,
            files={"file": ("test.tar.gz", tar_bytes.getvalue(), "application/gzip")},
        )
        assert res.status_code == 202


class TestArchiveDownload:
    def test_download_not_found(self, client, auth_header):
        res = client.get("/batch/archive-download/nonexistent", headers=auth_header)
        assert res.status_code == 404

    def test_download_not_completed(self, client, auth_header, isolated_env):
        store = isolated_env["store"]
        jid = store.create_job("test.zip", "/ext", "/res", 1)

        res = client.get(f"/batch/archive-download/{jid}", headers=auth_header)
        assert res.status_code == 409
        assert "not yet completed" in res.json()["error"].lower()

    def test_download_completed(self, client, auth_header, isolated_env):
        store = isolated_env["store"]
        uploads = isolated_env["uploads"]

        jid = store.create_job("test.zip", "/ext", "/res", 1)
        store.set_status(jid, "completed")

        # Create a fake result zip
        zip_path = os.path.join(uploads, "result.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", "[]")
        store.set_download_zip(jid, zip_path)

        res = client.get(f"/batch/archive-download/{jid}", headers=auth_header)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/zip"
