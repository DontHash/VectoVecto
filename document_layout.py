"""
document_layout.py — reading-order sorting for OCR tokens (XY-cut).

Why: OCR detection order is not reading order. PP-OCR tends to return boxes in
rows, so a two-column page comes out interleaved (left1, right1, left2, ...);
Tesseract gives per-line word order but no column model. Fixing order before
the transcript/searchable-PDF/JSON is a measurable CER/WER win on multi-column
pages and a no-op on single-column ones.

Algorithm (conservative XY-cut, measured against SROIE/CORD/arXiv renders):
  * a vertical whitespace band that at most 5% of token boxes cross, with
    substance on both sides, right side ragged (text column, not right-aligned
    numbers) and non-numeric -> that is a real gutter -> column-major within
    the region. Coverage-based, because real OCR line boxes extend into the
    gutter (clean union gaps of 8-52 px at 200 dpi vs a 2x-line-height rule
    that never fires); occasional straddlers (equations, figures) are assigned
    by center and guarded by the 5% limit.
  * full-width elements (titles, tables) split the page into vertical bands so
    each region gets its own column decision (a page-wide gutter does not
    exist on title + two-column layouts)
  * no confident gutter -> preserve engine order, which is already row-major
    for both supported backends (RapidOCR detection order, Tesseract lines);
    aggressive row re-grouping measurably HURTS single-column receipts

API:
    sort_reading_order(tokens) -> new list in reading order
"""
from __future__ import annotations

import statistics
from typing import List, Optional, Sequence, Tuple

import numpy as np

from document_ocr import Token

# A gutter x-run may be crossed by at most this share of token boxes (measured:
# arXiv renders 0-5%, SROIE receipts never produce a candidate at all).
COVERAGE_LIMIT = 0.05
# ... and must be at least this wide (px / x median line height).
GUTTER_MIN_WIDTH_PX = 8.0
GUTTER_MIN_WIDTH_FACTOR = 0.3
# The gutter must sit in the middle of the content (20%-80%) to count.
GUTTER_CENTER_MIN = 0.2
GUTTER_CENTER_MAX = 0.8
# Each side must be a wide text block, not a thin label strip: on receipts the
# item/label strip next to a value column produced a passable gutter (measured:
# sroie_00003 label strip was 24% of content width) and columnizing it
# reordered rows. Real text columns are ~40-50% of the content width; 0.3
# keeps 0/60 real single-column pages changed and all 19 arXiv splits.
SIDE_MIN_WIDTH_FRACTION = 0.3
# A column split needs substance on both sides: label/value tables (receipt
# totals, invoices) have clean gutters but only a few rows, and reading them
# column-major is wrong. Real columns have many lines per side.
MIN_X_SIDE_TOKENS = 6
# A token at least this fraction of the region's content width is a full-width
# element (title, section header): it always separates vertical bands. Without
# this, a title sitting ~1 line-height above the columns merges into the column
# band and its width shrinks the gutter below threshold (measured on the
# two-column fixture: pages 2-3 failed to columnize).
WIDE_TOKEN_FRACTION = 0.5
# Vertical bands: merge token y-intervals whose gap is below ~2.5 line heights.
BAND_GAP_FACTOR = 2.5
BAND_GAP_MIN_PX = 32.0
MAX_DEPTH = 16


def _find_gutter(tokens: Sequence[Token], med_h: float) -> Optional[Tuple[float, float]]:
    """Widest x-run covered by <= COVERAGE_LIMIT of token boxes, or None.

    Coverage (not emptiness) is what works on real pages: OCR line boxes are
    wider than the printed text, so a few px of box padding close an otherwise
    clean gutter; requiring zero crossings has no operating point.
    """
    lo = min(t.bbox[0] for t in tokens)
    hi = max(t.bbox[2] for t in tokens)
    cw = hi - lo
    if cw <= 0:
        return None
    hist = np.zeros(int(cw) + 2, dtype=np.int32)
    for t in tokens:
        a, b = t.bbox[0] - lo, t.bbox[2] - lo
        hist[max(0, a):max(0, a) + max(1, b - a)] += 1
    limit = max(1, int(COVERAGE_LIMIT * len(tokens)))
    min_w = max(GUTTER_MIN_WIDTH_PX, GUTTER_MIN_WIDTH_FACTOR * med_h)
    best: Optional[Tuple[float, float]] = None
    best_w = 0.0
    run: Optional[int] = None
    for i in range(len(hist) + 1):
        empty = i < len(hist) and hist[i] <= limit
        if empty:
            if run is None:
                run = i
            continue
        if run is None:
            continue
        w = i - run
        x0, x1 = run + lo, i + lo
        run = None
        if w < min_w:
            continue
        center = (x0 + x1) / 2.0
        if not (lo + GUTTER_CENTER_MIN * cw < center < lo + GUTTER_CENTER_MAX * cw):
            continue
        if w > best_w:
            best, best_w = (x0, x1), w
    return best


def _median_height(tokens: Sequence[Token]) -> float:
    heights = [max(1.0, t.bbox[3] - t.bbox[1]) for t in tokens]
    return max(1.0, statistics.median(heights))


def _ragged_right(tokens: Sequence[Token]) -> bool:
    """Does the right side read like a text column (ragged right) rather than a
    value column (right-aligned amounts)?

    Text columns are left-aligned with varying line lengths (sd(x1) >= sd(x0));
    receipt/invoice value columns are right-aligned (sd(x1) < sd(x0)), and
    reading them column-major is wrong. This is the geometric signal that keeps
    label/value tables row-major on SROIE while still columnizing real pages.
    """
    x0s = [float(t.bbox[0]) for t in tokens]
    x1s = [float(t.bbox[2]) for t in tokens]
    if len(tokens) < 2:
        return True
    sd0 = statistics.pstdev(x0s)
    sd1 = statistics.pstdev(x1s)
    return sd1 >= 0.5 * sd0


def _digit_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isdigit()) / len(chars)


def _mostly_numeric(tokens: Sequence[Token], threshold: float = 0.4) -> bool:
    """A number-dominated right side is a value column, not a text column.

    SROIE totals blocks have 'RM XX.XX' rows where x0 and x1 are both nearly
    constant (fixed currency prefix), so edge alignment cannot see the table.
    Content can: numbers read row-wise (label: value), never column-wise.
    """
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _digit_ratio(t.text) >= threshold)
    return numeric / len(tokens) >= 0.5


def row_major(tokens: Sequence[Token]) -> List[Token]:
    """Rows top-down (y-overlap grouping), left-right inside a row."""
    items = sorted(tokens, key=lambda t: (t.bbox[1] + t.bbox[3]) / 2.0)
    rows: List[dict] = []
    for tok in items:
        cy = (tok.bbox[1] + tok.bbox[3]) / 2.0
        th = max(1.0, tok.bbox[3] - tok.bbox[1])
        for row in rows:
            rcy = (row["y0"] + row["y1"]) / 2.0
            rh = max(1.0, row["y1"] - row["y0"])
            if abs(cy - rcy) <= 0.5 * max(th, rh):
                row["tokens"].append(tok)
                row["y0"] = min(row["y0"], tok.bbox[1])
                row["y1"] = max(row["y1"], tok.bbox[3])
                break
        else:
            rows.append({"y0": tok.bbox[1], "y1": tok.bbox[3], "tokens": [tok]})
    out: List[Token] = []
    for row in rows:
        out.extend(sorted(row["tokens"], key=lambda t: (t.bbox[0], t.bbox[1])))
    return out


def _content_width(tokens: Sequence[Token]) -> float:
    return max(t.bbox[2] for t in tokens) - min(t.bbox[0] for t in tokens)


def _y_overlap(a: Token, b: Token) -> float:
    return min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])


def _standalone_wide_ids(tokens: Sequence[Token], cw: float) -> set:
    """Full-width tokens that sit alone on their row are structural separators.

    A wide token sharing its row with other tokens is not (measured: receipt
    footer lines overlap the amounts column; treating them as separators
    reordered rows and cost 8% CER on SROIE).
    """
    if cw <= 0:
        return set()
    wide = [t for t in tokens if (t.bbox[2] - t.bbox[0]) >= WIDE_TOKEN_FRACTION * cw]
    ids = set()
    for t in wide:
        if not any(other is not t and _y_overlap(t, other) > 1.0 for other in tokens):
            ids.add(id(t))
    return ids


def _bands(tokens: Sequence[Token], med_h: float,
           wide_ids: Optional[set] = None) -> List[List[Token]]:
    """Vertical connected components: token y-intervals merged when the gap is
    below ~2.5 line heights; full-width tokens are forced separators so titles
    and tables get their own bands (each band judged for columns on its own)."""
    items = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[3]))
    gap_thresh = max(BAND_GAP_FACTOR * med_h, BAND_GAP_MIN_PX)
    bands: List[List[Token]] = []
    cur: List[Token] = []
    cur_y1 = 0.0
    for tok in items:
        if wide_ids and id(tok) in wide_ids:
            if cur:
                bands.append(cur)
                cur = []
            bands.append([tok])
            continue
        if not cur:
            cur = [tok]
            cur_y1 = tok.bbox[3]
        elif tok.bbox[1] - cur_y1 <= gap_thresh:
            cur.append(tok)
            cur_y1 = max(cur_y1, tok.bbox[3])
        else:
            bands.append(cur)
            cur = [tok]
            cur_y1 = tok.bbox[3]
    if cur:
        bands.append(cur)
    return bands


def _side_width(tokens: Sequence[Token]) -> float:
    return max(t.bbox[2] for t in tokens) - min(t.bbox[0] for t in tokens)


def _column_split(tokens: Sequence[Token], med_h: float, min_side: int,
                  stats: dict) -> Optional[List[Token]]:
    """A confident gutter -> left side sorted, then right side sorted."""
    gutter = _find_gutter(tokens, med_h)
    if gutter is None:
        return None
    x0, x1 = gutter
    mid = (x0 + x1) / 2.0
    left = [t for t in tokens if (t.bbox[0] + t.bbox[2]) / 2.0 < mid]
    right = [t for t in tokens if (t.bbox[0] + t.bbox[2]) / 2.0 >= mid]
    if len(left) < min_side or len(right) < min_side:
        return None
    if not _ragged_right(right) or _mostly_numeric(right):
        return None
    cw = _content_width(tokens)
    if (_side_width(left) < SIDE_MIN_WIDTH_FRACTION * cw
            or _side_width(right) < SIDE_MIN_WIDTH_FRACTION * cw):
        return None
    stats["splits"] += 1
    return (_sort_region(left, min_side, 1, stats)
            + _sort_region(right, min_side, 1, stats))


def _sort_region(tokens: Sequence[Token], min_side: int, depth: int = 0,
                 stats: Optional[dict] = None) -> List[Token]:
    """Conservative reading order: repair confirmed column layouts, otherwise
    preserve engine order (RapidOCR and Tesseract already emit rows top-down)."""
    if stats is None:
        stats = {"splits": 0}
    if len(tokens) <= 1 or depth >= MAX_DEPTH:
        return list(tokens)
    med_h = _median_height(tokens)

    split = _column_split(tokens, med_h, min_side, stats)
    if split is not None:
        return split

    cw = _content_width(tokens)
    wide_ids = _standalone_wide_ids(tokens, cw)
    if len(wide_ids) >= 0.5 * len(tokens):
        wide_ids = set()  # every token fills the measure: single-column region
    bands = _bands(tokens, med_h, wide_ids)
    if len(bands) <= 1:
        return list(tokens)
    out: List[Token] = []
    for band in bands:
        out.extend(_sort_region(band, min_side, depth + 1, stats))
    return out


def sort_reading_order(tokens: Sequence[Token],
                       min_column_tokens: int = MIN_X_SIDE_TOKENS,
                       stats: Optional[dict] = None) -> List[Token]:
    """Return tokens in human reading order (new list; input untouched).

    If no confident column split exists anywhere, the input order is returned
    unchanged: band bookkeeping must never perturb single-column pages.
    `stats["splits"]` (when a dict is passed) records how many column splits
    fired, for telemetry.
    """
    toks = list(tokens)
    if not toks:
        return []
    book = {"splits": 0}
    out = _sort_region(toks, max(1, min_column_tokens), 0, book)
    if stats is not None:
        stats["splits"] = book["splits"]
    return out if book["splits"] else list(tokens)


def text_in_order(tokens: Sequence[Token]) -> str:
    return "\n".join(t.text for t in tokens if t.text)


if __name__ == "__main__":
    from document_ocr import Token as T

    words = []
    for i in range(6):
        y = 40 + i * 80
        words.append(T(f"right column line {i + 1}", 99, (400, y, 700, y + 30), "line"))
        words.append(T(f"left column line {i + 1}", 99, (40, y, 300, y + 30), "line"))
    out = sort_reading_order(words)
    for t in out:
        print(t.text)
    expected = [f"left column line {i}" for i in range(1, 7)] + \
               [f"right column line {i}" for i in range(1, 7)]
    assert [t.text for t in out] == expected
    print("document_layout OK")
