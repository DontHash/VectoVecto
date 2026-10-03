"""
test_webapp_api.py — the FastAPI studio: health, upload validation, restore
plumbing, artifact serving, output toggles, multi-page runs, and server-side
failure logging.

The OCR pipeline itself is stubbed (it is covered by its own tests); these
tests exercise the web layer: request validation, run lifecycle, the PDF vs
image loader choice, option threading, the artifact whitelist and the payload.
"""
from __future__ import annotations

import io
import logging
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "webapp"))

pytest.importorskip("fastapi", reason="webapp dependencies not installed")
from fastapi.testclient import TestClient  # noqa: E402

import vvweb.api as api_mod  # noqa: E402
from vvweb.security import RateLimiter  # noqa: E402
from vvweb.storage import RunStore  # noqa: E402


def _png_bytes(size: int = 64) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "store", RunStore(str(tmp_path / "runs"), 60))
    monkeypatch.setattr(api_mod, "limiter", RateLimiter(120, 100))
    # no context manager: the lifespan warm-up is skipped in tests
    return TestClient(api_mod.create_app())


def _fake_pipeline(captured: dict | None = None):
    """Stand-in for run_document_pipeline: honours the make_* flags and writes
    the artifacts the web layer copies to canonical names."""
    import numpy as np

    from veriscript.document.ocr import OCRResult, Token

    class _FakeResult:
        def __init__(self):
            self.display_bgr = np.full((64, 64, 3), 255, np.uint8)
            self.ocr = OCRResult(text="HELLO", tokens=[
                Token(text="HELLO", conf=99, bbox=(1, 1, 30, 20),
                      flags=["low_conf"])],
                backend="fake")
            self.meta = {"seconds": 0.05, "backend": "fake",
                         "primary_stream": "raw", "skew_angle": 0.0,
                         "resized": False}
            self.status_line = "1 tokens"
            self.outputs = {}

    def fake(img, **kw):
        if captured is not None:
            captured.setdefault("calls", []).append(dict(kw))
        out_dir, stem = kw["out_dir"], kw["stem"]
        outputs = {}
        result = _FakeResult()
        if kw.get("make_txt", True):
            path = os.path.join(out_dir, f"{stem}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("HELLO\n")
            outputs["txt"] = path
        if kw.get("make_md", True):
            path = os.path.join(out_dir, f"{stem}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Transcript\n\nHELLO\n")
            outputs["md"] = path
        if kw.get("make_overlay", True):
            from PIL import Image
            path = os.path.join(out_dir, f"{stem}_overlay.png")
            Image.new("RGB", (8, 8), (255, 255, 255)).save(path)
            outputs["overlay"] = path
        if kw.get("make_pdf", True):
            path = os.path.join(out_dir, f"{stem}.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4 fake")
            outputs["pdf"] = path
        if kw.get("make_json", True):
            from veriscript.document.export import write_ocr_json
            path = os.path.join(out_dir, f"{stem}.json")
            write_ocr_json(path, result.ocr)
            outputs["json"] = path
        result.outputs = outputs
        return result

    return fake


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True and payload["version"]
    assert payload["limits"]["max_upload_mb"] > 0
    assert payload["limits"]["max_pages"] > 0
    assert payload["limits"]["daily_pages"] > 0
    assert payload["limits"]["pages_per_client"] > 0
    # hosting hardening
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in r.headers
    # small responses must not be gzipped (minimum-size rule holds even
    # through the real middleware stack)
    gz = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in gz.headers


def test_restore_rejects_unsupported_file(client):
    r = client.post("/api/restore",
                    files={"file": ("page.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "Unsupported file type" in r.text


def test_restore_rejects_an_empty_file(client):
    r = client.post("/api/restore",
                    files={"file": ("page.png", b"", "image/png")})
    assert r.status_code == 400
    assert "empty" in r.text.lower()


def test_restore_happy_path_serves_the_markdown(client, monkeypatch):
    from veriscript.document import pipeline as document_pipeline

    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline())
    r = client.post("/api/restore",
                    files={"file": ("page.png", _png_bytes(), "image/png")},
                    data={"lang": "en", "deskew": "0"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["files"]["md"].endswith("/files/transcript.md")

    dl = client.get(payload["files"]["md"])
    assert dl.status_code == 200
    assert "Transcript" in dl.text

    # the artifact whitelist still rejects anything else
    bad = client.get(payload["files"]["md"].replace("transcript.md",
                                                    "manifest.json"))
    assert bad.status_code in (404, 400)


def test_restore_threads_engine_and_options(client, monkeypatch):
    from veriscript.document import pipeline as document_pipeline

    captured: dict = {}
    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline(captured))
    r = client.post("/api/restore",
                    files={"file": ("page.png", _png_bytes(), "image/png")},
                    data={"ocr": "tesseract", "auto_rotate": "off",
                          "lang": "en", "deskew": "1"})
    assert r.status_code == 200, r.text
    call = captured["calls"][0]
    assert call["backend"] == "tesseract"
    assert call["auto_rotate"] is False
    assert call["deskew"] is True
    assert r.json()["meta"]["auto_rotate"] is False


def test_output_toggles_hide_artifacts(client, monkeypatch):
    from veriscript.document import pipeline as document_pipeline

    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline())
    r = client.post("/api/restore",
                    files={"file": ("page.png", _png_bytes(), "image/png")},
                    data={"pdf": "0", "md": "0", "overlay": "0"})
    assert r.status_code == 200, r.text
    files = r.json()["files"]
    assert "pdf" not in files and "md" not in files and "overlay" not in files
    assert "txt" in files, "the transcript fallback still serves the text"


def test_pdf_upload_is_loaded_as_a_pdf(client, tmp_path, monkeypatch):
    """Regression: streamed uploads have no extension; the validator's kind
    must decide the loader, not the filename on disk."""
    from veriscript.document import pipeline as document_pipeline
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "one.pdf"
    c = canvas.Canvas(str(pdf), pagesize=(240, 200))
    c.drawString(20, 170, "PDF PAGE")
    c.showPage()
    c.save()

    captured: dict = {}
    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline(captured))
    r = client.post("/api/restore",
                    files={"file": ("one.pdf", pdf.read_bytes(),
                                    "application/pdf")})
    assert r.status_code == 200, r.text
    assert captured["calls"][0]["backend"] is None
    payload = r.json()
    assert payload["meta"]["input_kind"] == "pdf"
    assert payload["meta"]["pages_processed"] == 1


def test_all_pages_writes_combined_outputs(client, tmp_path, monkeypatch):
    from veriscript.document import pipeline as document_pipeline
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "three.pdf"
    c = canvas.Canvas(str(pdf), pagesize=(240, 200))
    for i in range(3):
        c.drawString(20, 170, f"PAGE {i}")
        c.showPage()
    c.save()

    captured: dict = {}
    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline(captured))
    r = client.post("/api/restore",
                    files={"file": ("three.pdf", pdf.read_bytes(),
                                    "application/pdf")},
                    data={"all_pages": "1", "lang": "en"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["meta"]["pages_processed"] == 3
    assert len(captured["calls"]) == 3

    files = payload["files"]
    assert {"combined_pdf", "combined_txt", "combined_md"} <= set(files)
    md = client.get(files["combined_md"])
    assert md.status_code == 200
    assert "## Page 1" in md.text and "## Page 3" in md.text


def test_per_visitor_page_quota_blocks(client, monkeypatch):
    from veriscript.document import pipeline as document_pipeline
    from vvweb.security import DailyKeyedQuota

    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline())
    monkeypatch.setattr(api_mod, "client_pages", DailyKeyedQuota(2))

    for _ in range(2):
        r = client.post("/api/restore",
                        files={"file": ("page.png", _png_bytes(), "image/png")},
                        data={"lang": "en"})
        assert r.status_code == 200, r.text

    r = client.post("/api/restore",
                    files={"file": ("page.png", _png_bytes(), "image/png")},
                    data={"lang": "en"})
    assert r.status_code == 429
    assert "free demo limit" in r.text
    assert r.headers.get("retry-after")


def test_pdf_pages_are_charged_to_the_quota(client, tmp_path, monkeypatch):
    """A 2-page PDF costs 2 pages of the visitor's daily budget."""
    from reportlab.pdfgen import canvas
    from vvweb.security import DailyKeyedQuota

    pdf = tmp_path / "two.pdf"
    c = canvas.Canvas(str(pdf), pagesize=(240, 200))
    for i in range(2):
        c.drawString(20, 170, f"PAGE {i}")
        c.showPage()
    c.save()

    monkeypatch.setattr(api_mod, "client_pages", DailyKeyedQuota(1))
    r = client.post("/api/restore",
                    files={"file": ("two.pdf", pdf.read_bytes(),
                                    "application/pdf")},
                    data={"all_pages": "1", "lang": "en"})
    assert r.status_code == 429
    assert "free demo limit" in r.text


def test_restore_failure_is_logged(client, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(api_mod, "process_page", boom)
    with caplog.at_level(logging.ERROR, logger="vectovecto.web"):
        r = client.post("/api/restore",
                        files={"file": ("page.png", _png_bytes(), "image/png")})

    assert r.status_code == 500
    errors = [rec for rec in caplog.records
              if rec.name == "vectovecto.web" and rec.levelno == logging.ERROR]
    assert errors, "the server side must log the failure"
    assert errors[0].exc_info and "pipeline exploded" in str(errors[0].exc_info[1])


# ---------------------------------------------------------------------------
# correction loop (docs/CORRECTIONS.md)
# ---------------------------------------------------------------------------

def _restore_once(client, monkeypatch, data=None):
    """A finished run with the fake pipeline (canonical artifacts, ocr.json)."""
    from veriscript.document import pipeline as document_pipeline

    monkeypatch.setattr(document_pipeline, "run_document_pipeline",
                        _fake_pipeline())
    r = client.post("/api/restore",
                    files={"file": ("page.png", _png_bytes(), "image/png")},
                    data={"lang": "en", **(data or {})})
    assert r.status_code == 200, r.text
    return r.json()


def test_correct_applies_and_serves_corrected_files(client, monkeypatch):
    payload = _restore_once(client, monkeypatch)
    r = client.post(f"/api/runs/{payload['run_id']}/correct", json={
        "corrections": [{"index": 0, "bbox": [1, 1, 30, 20],
                         "original": "HELLO", "corrected": "HELLO2"}]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stats"]["changed"] == 1
    assert out["review"] == []
    # the corrected record is the truth once it exists
    assert out["transcript"] == "HELLO2\n"
    for key in ("corrected_pdf", "corrected_txt", "corrected_json",
                "corrections_json"):
        assert key in out["files"], key

    corrected = client.get(out["files"]["corrected_json"]).json()
    entry = corrected["tokens"][0]
    assert entry["text"] == "HELLO2"
    assert entry["original_text"] == "HELLO"
    assert entry["text_source"] == "human" and entry["corrected_by"] == "human"

    # the originals are never overwritten
    original = client.get(out["files"]["json"]).json()
    assert original["tokens"][0]["text"] == "HELLO"
    assert original["tokens"][0]["original_text"] is None


def test_correct_is_cumulative_and_memory_records_once(client, monkeypatch,
                                                       tmp_path):
    """A new batch builds on the run's corrected state; memory never dupes."""
    monkeypatch.setenv("VERISCRIPT_MEMORY", "1")
    monkeypatch.setenv("VERISCRIPT_MEMORY_DIR", str(tmp_path / "mem.jsonl"))
    from veriscript.document import memory

    payload = _restore_once(client, monkeypatch)
    run = payload["run_id"]
    first = {"index": 0, "bbox": [1, 1, 30, 20], "original": "HELLO",
             "corrected": "HELLO2"}
    r = client.post(f"/api/runs/{run}/correct", json={"corrections": [first]})
    assert r.status_code == 200 and r.json()["stats"]["changed"] == 1
    assert sum(1 for e in memory._iter_events()
               if e.get("kind") == "correction") == 1

    # Re-posting an already-applied batch is idempotent and not re-recorded.
    r = client.post(f"/api/runs/{run}/correct", json={"corrections": [first]})
    assert r.status_code == 200 and r.json()["review"] == []
    assert sum(1 for e in memory._iter_events()
               if e.get("kind") == "correction") == 1

    # A new correction applies on top (cumulative corrected.json), reports
    # only itself, and lands in memory once.
    second = {"index": 0, "bbox": [1, 1, 30, 20], "original": "HELLO2",
              "corrected": "HELLO3"}
    r = client.post(f"/api/runs/{run}/correct", json={"corrections": [second]})
    assert r.status_code == 200
    out = r.json()
    assert out["stats"]["changed"] == 1 and out["stats"]["reviewed"] == 1
    corrected = client.get(out["files"]["corrected_json"]).json()
    assert corrected["tokens"][0]["text"] == "HELLO3"
    assert sum(1 for e in memory._iter_events()
               if e.get("kind") == "correction") == 2


def test_correct_reports_stale_entries_as_skipped(client, monkeypatch):
    payload = _restore_once(client, monkeypatch)
    r = client.post(f"/api/runs/{payload['run_id']}/correct", json={
        "corrections": [{"index": 0, "bbox": [9000, 9000, 9100, 9040],
                         "original": "SOMETHING ELSE", "corrected": "X"}]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stats"]["skipped"] == 1 and out["stats"]["reviewed"] == 0
    assert "corrected_json" not in out["files"]


def test_correct_rejects_malformed_bodies(client, monkeypatch):
    payload = _restore_once(client, monkeypatch)
    run = payload["run_id"]
    assert client.post(f"/api/runs/{run}/correct",
                       json={"corrections": [{"corrected": "no index"}]}
                       ).status_code == 400
    assert client.post(f"/api/runs/{run}/correct",
                       json={"corrections": []}).status_code == 400
    assert client.post("/api/runs/0000000000000000/correct",
                       json={"corrections": [{"index": 0, "corrected": "x"}]}
                       ).status_code == 404


def test_corrections_export_contains_only_shared_crops(client, monkeypatch):
    import io
    import zipfile

    payload = _restore_once(client, monkeypatch)
    body = {"corrections": [
        {"index": 0, "bbox": [1, 1, 30, 20], "original": "HELLO",
         "corrected": "HELLO2"}],
        "share": [0]}
    r = client.post(f"/api/runs/{payload['run_id']}/corrections/export",
                    json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["shared"] == 1
    z = client.get(out["file"])
    assert z.status_code == 200
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        names = set(zf.namelist())
        assert {"README.txt", "corrections.json", "labels.tsv"} <= names
        assert any(n.startswith("crops/") for n in names)
        assert "HELLO2" in zf.read("labels.tsv").decode("utf-8")

    # nothing marked -> no crops, still an explicit file
    body["share"] = []
    r2 = client.post(f"/api/runs/{payload['run_id']}/corrections/export",
                     json=body)
    assert r2.status_code == 200
    assert r2.json()["shared"] == 0
    z2 = client.get(r2.json()["file"])
    with zipfile.ZipFile(io.BytesIO(z2.content)) as zf:
        assert not [n for n in zf.namelist() if n.startswith("crops/")]


def test_corrections_export_rejects_unknown_share(client, monkeypatch):
    payload = _restore_once(client, monkeypatch)
    r = client.post(f"/api/runs/{payload['run_id']}/corrections/export",
                    json={"corrections": [], "share": [7]})
    assert r.status_code == 400


def test_collect_run_summary_passes_suggestions(tmp_path):
    """Harness evidence is additive: the queue payload carries it through."""
    import json

    from vvweb.pipeline import collect_run_summary

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ocr.json").write_text(json.dumps({
        "review": [{"index": 0, "text": "१०", "flags": ["digit_uncertain"],
                    "suggestions": [{"text": "२०", "source": "memory",
                                     "why": "corrected 2x before"}]}],
        "tokens": [{"text": "१०"}],
    }), encoding="utf-8")

    out = collect_run_summary(str(run_dir), "a" * 16)
    assert out["review"][0]["suggestions"][0]["source"] == "memory"


def test_memory_suggestions_feedback_and_clear(client, monkeypatch, tmp_path):
    """Track D2 end to end: suggest -> accept -> feedback -> clear."""
    monkeypatch.setenv("VERISCRIPT_MEMORY", "1")
    monkeypatch.setenv("VERISCRIPT_MEMORY_DIR", str(tmp_path / "mem.jsonl"))
    from veriscript.document import memory

    memory.record_corrections(
        [{"index": 0, "corrected": "HELLO2", "action": "changed"}],
        {0: ("HELLO", ["low_conf"])})

    payload = _restore_once(client, monkeypatch)
    suggestion = payload["review"][0]["suggestions"][0]
    assert suggestion["text"] == "HELLO2" and suggestion["source"] == "memory"

    r = client.post(f"/api/runs/{payload['run_id']}/correct", json={
        "corrections": [{"index": 0, "bbox": [1, 1, 30, 20], "original": "HELLO",
                         "corrected": "HELLO2", "suggested": "HELLO2",
                         "suggestion_source": "memory"}]})
    assert r.status_code == 200, r.text
    assert any(e.get("kind") == "accept" for e in memory._iter_events())

    assert client.get("/api/health").json()["limits"]["memory"] is True
    cleared = client.post("/api/memory/clear")
    assert cleared.status_code == 200 and cleared.json()["removed"] >= 1
    assert not os.path.exists(str(tmp_path / "mem.jsonl"))


def test_memory_off_by_default(client, monkeypatch):
    monkeypatch.delenv("VERISCRIPT_MEMORY", raising=False)
    assert client.get("/api/health").json()["limits"]["memory"] is False
