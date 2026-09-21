"""
test_anchor_gemini.py — Gemini transcription for the anchor (N5b).

Vertex AI (ADC) transcribes the v2 pages; the local engines corroborate; the
audit worksheet shows Gemini vs the engines with disagreements marked. The
client is injected so tests never touch the network.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from anchor_gemini import run, strip_images, transcribe_page  # noqa: E402


class _Resp:
    def __init__(self, text):
        self.text = text


class _Models:
    def __init__(self, text="नमस्ते संसार", fail_first=0):
        self.calls = []
        self._text = text
        self._fail_first = fail_first

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents})
        if self._fail_first > 0:
            self._fail_first -= 1
            raise RuntimeError("transient")
        return _Resp(self._text)


class _Client:
    def __init__(self, text="नमस्ते संसार", fail_first=0):
        self.models = _Models(text, fail_first)


def test_strip_images_covers_page():
    img = np.zeros((1000, 400, 3), dtype=np.uint8)
    strips = strip_images(img, strips=2, overlap=40)
    assert len(strips) == 2
    assert strips[0].shape[0] == 540
    assert strips[-1].shape[1] == 400
    assert strips[0][0].sum() == 0 and strips[-1][-1].sum() == 0


def test_transcribe_page_calls_once_per_strip_and_joins():
    client = _Client("प्रथम पंक्ति")
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    text = transcribe_page(client, img, model="gemini-test", strips=2)
    assert len(client.models.calls) == 2
    assert text.count("प्रथम पंक्ति") == 2
    assert all(c["model"] == "gemini-test" for c in client.models.calls)


def test_transcribe_page_single_strip():
    client = _Client("एक")
    text = transcribe_page(client, np.zeros((300, 200, 3), dtype=np.uint8),
                           model="m", strips=1)
    assert len(client.models.calls) == 1
    assert text == "एक"


def test_transcribe_page_retries_transient_failures():
    client = _Client("ठीक", fail_first=1)
    text = transcribe_page(client, np.zeros((200, 120, 3), dtype=np.uint8),
                           model="m", strips=1, retries=2)
    assert text == "ठीक"
    assert len(client.models.calls) == 2


def test_transcribe_page_raises_after_retries():
    import pytest

    client = _Client(fail_first=99)
    with pytest.raises(RuntimeError):
        transcribe_page(client, np.zeros((200, 120, 3), dtype=np.uint8),
                        model="m", strips=1, retries=2)
    assert len(client.models.calls) == 3, "1 try + 2 retries"


def test_run_resume_reuses_existing_text_without_calling(tmp_path):
    import doc_data

    pdf = str(tmp_path / "demo.pdf")
    doc_data.make_demo_pdf(pdf)
    data_dir = str(tmp_path / "ds")
    manifest = doc_data.build_pdf_dataset([pdf], data_dir, dpi=72,
                                          degrade=False, gt_validity=True)
    page_id = manifest["entries"][0]["id"]
    out_dir = str(tmp_path / "anchor")
    verified = tmp_path / "anchor" / "verified"
    verified.mkdir(parents=True)
    (verified / f"{page_id}.txt").write_text("पहिले नै लेखिएको",
                                             encoding="utf-8")
    client = _Client(fail_first=99)  # any call would fail
    result = run(data_dir, out_dir, readings_path=None, model="m", strips=1,
                 client=client, resume=True)
    assert len(client.models.calls) == 0, "resume must not re-call the API"
    assert result["rows"][0]["readings"]["gemini"] == "पहिले नै लेखिएको"


def test_run_records_failures_and_continues(tmp_path):
    import doc_data

    pdf = str(tmp_path / "two.pdf")
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(pdf)
    for i in range(2):
        c.drawString(72, 720, f"PAGE {i}")
        c.showPage()
    c.save()
    data_dir = str(tmp_path / "ds")
    doc_data.build_pdf_dataset([pdf], data_dir, dpi=72, degrade=False,
                               gt_validity=True)
    client = _Client("ठीक", fail_first=1)  # first page fails, second succeeds
    result = run(data_dir, str(tmp_path / "anchor"), readings_path=None,
                 model="m", strips=1, client=client, retries=0)
    assert len(result["failed"]) == 1
    assert len(result["rows"]) == 1, "a failed page must not stop the run"


def test_run_writes_verified_files_and_readings(tmp_path):
    import doc_data

    pdf = str(tmp_path / "demo.pdf")
    doc_data.make_demo_pdf(pdf)
    data_dir = str(tmp_path / "ds")
    doc_data.build_pdf_dataset([pdf], data_dir, dpi=72, degrade=False,
                               gt_validity=True)
    out_dir = str(tmp_path / "anchor")
    client = _Client("INVOICE 1200.00")
    result = run(data_dir, out_dir, readings_path=None, model="gemini-test",
                 strips=1, client=client)
    page_id = result["rows"][0]["page"]
    verified = os.path.join(out_dir, "verified", f"{page_id}.txt")
    assert os.path.exists(verified)
    assert "INVOICE" in open(verified, encoding="utf-8").read()
    payload = json.load(open(os.path.join(out_dir, "gemini_readings.json"),
                             encoding="utf-8"))
    assert payload["worksheet"] == [page_id], "score() must accept all pages"
    assert payload["rows"][0]["readings"]["gemini"].startswith("INVOICE")
    assert os.path.exists(os.path.join(out_dir, "audit.html"))
