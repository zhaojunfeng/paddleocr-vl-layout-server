"""Tests for the AI Studio hosted API mode (AI_STUDIO_TOKEN)."""

import base64
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import server
from server import LayoutParsingRequest

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def _page(pruned, text, img_url):
    return {
        "prunedResult": pruned,
        "markdown": {"text": text, "images": {}},
        "outputImages": {},
        "inputImage": img_url,
    }


class TestOptionalPayload:
    def test_only_non_none_fields(self):
        req = LayoutParsingRequest(
            file="dGVzdA==", fileType=1,
            useDocUnwarping=True,
            markdownIgnoreLabels=["header", "footer"],
        )
        payload = server._build_aistudio_optional_payload(req)
        assert payload["useDocUnwarping"] is True
        assert payload["markdownIgnoreLabels"] == ["header", "footer"]
        # request default values are included (they mirror AI Studio defaults)
        assert payload["useLayoutDetection"] is True
        assert payload["temperature"] == 0

    def test_requires_file(self):
        with pytest.raises(ValueError):
            LayoutParsingRequest(file=None, filePath=None)


class TestResolveImageValue:
    def test_data_url_kept(self):
        assert server._resolve_image_value("data:image/jpeg;base64,AAAA") == \
            "data:image/jpeg;base64,AAAA"

    def test_http_url_converted_to_data_url(self):
        resp = SimpleNamespace(
            status_code=200, content=JPEG_BYTES,
            headers={"Content-Type": "image/jpeg"},
            raise_for_status=lambda: None,
        )
        with patch.object(server.requests, "get", return_value=resp) as m:
            out = server._resolve_image_value("http://cdn.example.com/img1.jpg")
        m.assert_called_once()
        assert out.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(out.split(",", 1)[1]) == JPEG_BYTES


class TestMergeResults:
    def test_single_image(self):
        lines = [{
            "result": {
                "layoutParsingResults": [_page(
                    {"width": 800, "height": 600}, "hello", "http://cdn/i.jpg")],
                "preprocessedImages": ["http://cdn/pp.jpg"],
                "dataInfo": {"type": "image", "width": 800, "height": 600},
            }
        }]
        with patch.object(server.requests, "get", return_value=SimpleNamespace(
            status_code=200, content=JPEG_BYTES,
            headers={"Content-Type": "image/jpeg"},
            raise_for_status=lambda: None,
        )):
            merged = server._merge_aistudio_results(lines, is_pdf=False)

        assert len(merged["layoutParsingResults"]) == 1
        assert merged["dataInfo"]["type"] == "image"
        assert merged["dataInfo"]["width"] == 800
        assert merged["dataInfo"]["height"] == 600
        assert merged["preprocessedImages"][0].startswith("data:image/jpeg;base64,")

    def test_multi_page_pdf(self):
        lines = [
            {"result": {
                "layoutParsingResults": [_page({"width": 100, "height": 200}, "p1",
                                                "http://cdn/p1.jpg")],
                "preprocessedImages": ["http://cdn/pp1.jpg"],
                "dataInfo": {"type": "pdf", "width": 100, "height": 200},
            }},
            {"result": {
                "layoutParsingResults": [_page({"width": 120, "height": 220}, "p2",
                                                "http://cdn/p2.jpg")],
                "preprocessedImages": ["http://cdn/pp2.jpg"],
                "dataInfo": {"type": "pdf", "width": 120, "height": 220},
            }},
        ]
        with patch.object(server.requests, "get", return_value=SimpleNamespace(
            status_code=200, content=JPEG_BYTES,
            headers={"Content-Type": "image/jpeg"},
            raise_for_status=lambda: None,
        )):
            merged = server._merge_aistudio_results(lines, is_pdf=True)

        assert len(merged["layoutParsingResults"]) == 2
        assert merged["dataInfo"]["numPages"] == 2
        assert merged["dataInfo"]["pages"] == [
            {"width": 100, "height": 200}, {"width": 120, "height": 220}]
        md = "".join(p["markdown"]["text"] for p in merged["layoutParsingResults"])
        assert md == "p1p2"

    def test_multi_page_without_data_info_falls_back_to_pruned(self):
        lines = [
            {"result": {"layoutParsingResults": [
                _page({"width": 10, "height": 20}, "p1", "")]}},
            {"result": {"layoutParsingResults": [
                _page({"width": 30, "height": 40}, "p2", "")]}},
        ]
        merged = server._merge_aistudio_results(lines, is_pdf=True)
        assert merged["dataInfo"]["pages"] == [
            {"width": 10, "height": 20}, {"width": 30, "height": 40}]


class TestProcessWithAistudio:
    def _fake_get(self, jsonl_text):
        jsonl_url = "https://result.example.com/out.jsonl"
        job_url = f"{server.AI_STUDIO_JOB_URL}/job123"

        def fake_get(url, headers=None, timeout=None):
            if url == jsonl_url:
                return SimpleNamespace(
                    status_code=200, text=jsonl_text,
                    raise_for_status=lambda: None,
                )
            if url == job_url:
                return SimpleNamespace(status_code=200, json=lambda: {
                    "data": {"state": "done", "extractProgress": {
                        "totalPages": 2, "extractedPages": 2,
                        "startTime": 1, "endTime": 2,
                    }, "resultUrl": {"jsonUrl": jsonl_url}},
                })
            # image download
            return SimpleNamespace(
                status_code=200, content=JPEG_BYTES,
                headers={"Content-Type": "image/jpeg"},
                raise_for_status=lambda: None,
            )
        return fake_get

    def test_full_flow(self, tmp_path, monkeypatch):
        jsonl = "\n".join([
            json.dumps({"result": {"layoutParsingResults": [
                _page({"width": 100, "height": 200}, "p1", "http://cdn/p1.jpg")],
                "preprocessedImages": ["http://cdn/pp1.jpg"],
                "dataInfo": {"type": "pdf", "width": 100, "height": 200}}}),
            json.dumps({"result": {"layoutParsingResults": [
                _page({"width": 120, "height": 220}, "p2", "http://cdn/p2.jpg")],
                "preprocessedImages": ["http://cdn/pp2.jpg"],
                "dataInfo": {"type": "pdf", "width": 120, "height": 220}}}),
        ])
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "test-token")

        post_resp = SimpleNamespace(
            status_code=200, json=lambda: {"data": {"jobId": "job123"}},
        )
        with patch.object(server.requests, "post", return_value=post_resp) as m_post, \
             patch.object(server.requests, "get", side_effect=self._fake_get(jsonl)) as m_get:
            file_path = tmp_path / "doc.pdf"
            file_path.write_bytes(b"%PDF-1.4 fake")
            result = server.process_with_aistudio(
                str(file_path), True,
                LayoutParsingRequest(filePath=str(file_path)),
                log_id="test-log",
            )

        # submit: one POST with multipart file
        assert m_post.call_count == 1
        submit_kwargs = m_post.call_args.kwargs
        assert submit_kwargs["data"]["model"] == server.AI_STUDIO_MODEL
        assert json.loads(submit_kwargs["data"]["optionalPayload"])["useLayoutDetection"] is True

        assert result["logId"] == "test-log"
        assert result["errorCode"] == 0
        res = result["result"]
        assert len(res["layoutParsingResults"]) == 2
        assert res["dataInfo"]["type"] == "pdf"
        assert res["dataInfo"]["numPages"] == 2
        assert res["preprocessedImages"][0].startswith("data:image/jpeg;base64,")
        assert res["layoutParsingResults"][0]["inputImage"].startswith("data:image/jpeg;base64,")

    def test_job_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "test-token")
        post_resp = SimpleNamespace(
            status_code=200, json=lambda: {"data": {"jobId": "bad"}},
        )

        def fake_get(url, headers=None, timeout=None):
            return SimpleNamespace(status_code=200, json=lambda: {
                "data": {"state": "failed", "errorMsg": "model crash"}})

        with patch.object(server.requests, "post", return_value=post_resp), \
             patch.object(server.requests, "get", side_effect=fake_get):
            file_path = tmp_path / "doc.pdf"
            file_path.write_bytes(b"%PDF-1.4 fake")
            with pytest.raises(RuntimeError, match="model crash"):
                server.process_with_aistudio(
                    str(file_path), True,
                    LayoutParsingRequest(filePath=str(file_path)),
                )

    def test_no_token_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "")
        file_path = tmp_path / "doc.pdf"
        file_path.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(RuntimeError, match="AI_STUDIO_TOKEN"):
            server.process_with_aistudio(
                str(file_path), True, LayoutParsingRequest(filePath=str(file_path)),
            )


class TestDispatch:
    def test_image_goes_to_aistudio_when_token_set(self, tmp_path, monkeypatch):
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "test-token")
        expected = {"logId": "x", "errorCode": 0, "errorMsg": "Success",
                    "result": {"layoutParsingResults": [], "preprocessedImages": [],
                               "dataInfo": {"type": "image"}}}

        with patch.object(server, "process_with_aistudio",
                          return_value=expected) as m_aistudio:
            out = server.process_single_file(
                LayoutParsingRequest(filePath=str(img_path)), log_id="log1")

        m_aistudio.assert_called_once()
        assert out == expected

    def test_scanned_pdf_goes_to_aistudio(self, tmp_path, monkeypatch):
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "test-token")
        monkeypatch.setattr(server, "is_scanned_pdf", lambda p: True)
        expected = {"logId": "x", "errorCode": 0, "errorMsg": "Success",
                    "result": {"layoutParsingResults": [], "preprocessedImages": [],
                               "dataInfo": {"type": "pdf"}}}

        with patch.object(server, "process_with_aistudio",
                          return_value=expected) as m_aistudio:
            out = server.process_single_file(
                LayoutParsingRequest(filePath=str(pdf_path)), log_id="log2")

        m_aistudio.assert_called_once()
        assert out == expected

    def test_local_pipeline_when_no_token(self, tmp_path, monkeypatch):
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        monkeypatch.setattr(server, "AI_STUDIO_TOKEN", "")

        fake_pipeline = MagicMock()
        fake_pipeline.predict.return_value = []
        with patch.object(server, "get_pipeline", return_value=fake_pipeline), \
             patch.object(server, "load_pages_from_file", return_value={}), \
             patch.object(server, "process_with_aistudio",
                          side_effect=AssertionError("should not be called")) as m_ai:
            out = server.process_single_file(
                LayoutParsingRequest(filePath=str(img_path)), log_id="log3")

        m_ai.assert_not_called()
        assert out["errorCode"] == 0
        assert out["result"]["dataInfo"]["type"] == "image"

    def test_sniff_suffix(self):
        assert server._sniff_file_suffix(b"%PDF-1.4") == ".pdf"
        assert server._sniff_file_suffix(b"\xff\xd8\xff\xe0") == ".jpg"
        assert server._sniff_file_suffix(b"\x89PNG\r\n\x1a\n") == ".png"
        assert server._sniff_file_suffix(b"unknown") == ".png"
