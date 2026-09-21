"""
test_doc_metrics_boxes.py — detection box matching + ALTO parsing.
"""
from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from doc_data import build_heidata_dataset, parse_alto_page  # noqa: E402
from doc_metrics import box_match_report  # noqa: E402

ALTO_V4 = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="Page1" HEIGHT="1000" WIDTH="800">
      <PrintSpace HEIGHT="1000" WIDTH="800" VPOS="0" HPOS="0">
        <TextBlock HEIGHT="500" WIDTH="700" VPOS="100" HPOS="50">
          <TextLine HEIGHT="40" WIDTH="300" VPOS="120" HPOS="60">
            <String HEIGHT="40" WIDTH="300" VPOS="120" HPOS="60" CONTENT="प्रथम पंक्ति"/>
          </TextLine>
          <TextLine HEIGHT="40" WIDTH="250" VPOS="200" HPOS="60">
            <String HEIGHT="40" WIDTH="250" VPOS="200" HPOS="60" CONTENT="द्वितीय"/>
          </TextLine>
          <TextLine HEIGHT="40" WIDTH="0" VPOS="300" HPOS="60">
            <String HEIGHT="40" WIDTH="0" VPOS="300" HPOS="60" CONTENT=""/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


def test_parse_alto_page_extracts_lines_and_skips_empty():
    parsed = parse_alto_page(ALTO_V4.encode("utf-8"))
    assert parsed["width"] == 800 and parsed["height"] == 1000
    assert [l["text"] for l in parsed["lines"]] == ["प्रथम पंक्ति", "द्वितीय"]
    assert parsed["lines"][0]["bbox"] == [60, 120, 360, 160]


def test_parse_alto_page_tolerates_garbage():
    parsed = parse_alto_page(b"<not-alto/>")
    assert parsed == {"width": 0, "height": 0, "lines": []}


def test_box_match_report_perfect_and_partial():
    gts = [{"bbox": [0, 0, 100, 20]}, {"bbox": [0, 30, 100, 50]}]
    perfect = [[0, 0, 100, 20], [0, 30, 100, 50]]
    rep = box_match_report(gts, perfect)
    assert rep["coverage"] == 1.0 and rep["extra_rate"] == 0.0
    assert rep["mean_iou"] > 0.99

    # one box matched, one detection far away added
    hyp = [[0, 0, 100, 20], [500, 500, 600, 520]]
    rep2 = box_match_report(gts, hyp)
    assert rep2["coverage"] == 0.5
    assert rep2["matched"] == 1
    assert rep2["extra"] == 1 and rep2["extra_rate"] == 0.5

    # boxes below IoU threshold do not match
    rep3 = box_match_report([[0, 0, 100, 20]], [[50, 0, 150, 20]])
    assert rep3["coverage"] == 0.0 and rep3["extra_rate"] == 1.0


def test_box_match_report_counts_extras():
    rep = box_match_report([[0, 0, 100, 20]], [[0, 0, 100, 20], [200, 200, 240, 220]])
    assert rep["matched"] == 1
    assert rep["extra"] == 1
    assert rep["extra_rate"] == 0.5
    assert rep["gt_total"] == 1 and rep["hyp_total"] == 2


def test_box_match_report_area_view_handles_line_word_granularity():
    # GT line box, hyp split into three word boxes: IoU greedy scores 0,
    # area view must see full coverage with no extra centers.
    gt = [{"bbox": [0, 0, 300, 40]}]
    hyps = [[0, 0, 100, 40], [100, 0, 200, 40], [200, 0, 300, 40]]
    rep = box_match_report(gt, hyps, image_size=(300, 40))
    assert rep["coverage"] == 0.0, "no single hyp box reaches IoU 0.5"
    assert rep["gt_area_coverage"] == 1.0
    assert rep["hyp_center_extra_rate"] == 0.0

    # a detection whose center is outside every GT box counts as extra
    rep2 = box_match_report(gt, hyps + [[400, 0, 500, 40]], image_size=(500, 40))
    assert rep2["hyp_center_extra_rate"] == 0.25


def test_build_heidata_dataset_end_to_end(tmp_path):
    import zipfile

    zips = tmp_path / "zips"
    zips.mkdir()
    with zipfile.ZipFile(zips / "bookX.zip", "w") as z:
        z.writestr("bookX/bookX/alto/00.xml", ALTO_V4)
        import numpy as np
        import cv2
        img = np.full((1000, 800, 3), 240, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        z.writestr("bookX/bookX/00.jpg", buf.tobytes())

    out = tmp_path / "heidata_out"
    m = build_heidata_dataset(str(zips), str(out))
    assert m["kind"] == "real_pages"
    assert len(m["entries"]) == 1
    e = m["entries"][0]
    assert e["id"] == "bookX_p00" and e["n_lines"] == 2
    gt = open(os.path.join(out, e["gt"]), encoding="utf-8").read()
    assert "प्रथम पंक्ति" in gt
    boxes = json.load(open(os.path.join(out, e["boxes"]), encoding="utf-8"))
    assert boxes[1]["bbox"] == [60, 200, 310, 240]
    from doc_data import load_dataset
    loaded = load_dataset(str(out))
    assert "_boxes_path" in loaded["entries"][0]
