"""Digit-confusion candidates for the review queue (suggestion-only).

When the readers agree on a digit but the human is checking it anyway, the
confusion pairs below offer the alternative readings as one-click chips.
Nothing here is a flag: no text, conf or queue order changes, so frozen
metrics cannot move; the chips only enrich tokens already in the queue.

The table is measured, not guessed (2026-10-03, `nepali_pdf_v2` closest-GT
classification of 154 counted digit errors + the Appendix AH date cases):

  * 9 -> 1 (the dominant pair: dates and section numbers)
  * 9 -> 2 (Appendix AH date substitutions, e.g. ३९ -> ३२)
  * 9 -> 0 (the other AH date reading, ३९ -> ३०)
  * 3 -> 1, 2 -> 1 (section/list references)
  * 0 -> 6, 0 -> 8, 4 -> 2 (singletons, kept for coverage)

Most counted errors are not substitutions at all (55% are length /
segmentation artifacts), which bounds this layer's reach: it is a small,
precise add-on to human review, not a recognition fix. Measured on the full
`nepali_pdf_v2` set: chips are low-yield (tiered ranking, context-supported
first: 8.2% of chips in GT, ~2 error tokens covered per 41 pages), which is
why the layer stays opt-in (`--digit-confusions`) and off by default.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SOURCE = "confusion"

# Engine digit -> candidate digits, ordered by measured frequency.
DIGIT_PARTNERS: Dict[str, Tuple[str, ...]] = {
    "9": ("1", "2", "0"),
    "3": ("1",),
    "2": ("1",),
    "0": ("6", "8"),
    "4": ("2",),
}

# Flattened in measured-frequency order: the strongest confusion generates the
# first chip regardless of where it sits in the token.
_PARTNER_ORDER: Tuple[Tuple[str, str], ...] = tuple(
    (digit, partner)
    for digit, partners in DIGIT_PARTNERS.items()
    for partner in partners
)

_DEVA_DIGITS = "०१२३४५६७८९"
_TO_ASCII = {c: str(i) for i, c in enumerate(_DEVA_DIGITS)}
_TO_DEVA = {str(i): c for i, c in enumerate(_DEVA_DIGITS)}


def _digit_span(text: str) -> List[Tuple[int, str]]:
    """(index, ASCII digit) for every digit character in `text`."""
    out: List[Tuple[int, str]] = []
    for i, ch in enumerate(text or ""):
        if ch in _TO_ASCII:
            out.append((i, _TO_ASCII[ch]))
        elif ch.isdigit():
            out.append((i, ch))
    return out


def _ascii_digits(text: str) -> str:
    return "".join(_TO_ASCII.get(ch, ch) for ch in (text or "")
                   if ch in _TO_ASCII or ch.isdigit())


def generate(text: str, limit: int = 2,
             page_values: Optional[set] = None) -> List[Dict]:
    """Candidate readings that change one digit to a measured confusion.

    Deterministic. Candidates whose value already appears elsewhere on the
    page (`page_values`, ASCII digits) rank first - repetition is evidence -
    then measured pair frequency decides, so a 9->1 chip outranks a 2->1 chip
    wherever they sit. At most `limit` chips, no duplicates, never the input.
    """
    spans = _digit_span(text)
    candidates = []
    for order, (source, partner) in enumerate(_PARTNER_ORDER):
        for index, digit in spans:
            if digit != source:
                continue
            replacement = (_TO_DEVA.get(partner, partner)
                           if text[index] in _TO_ASCII else partner)
            candidate = text[:index] + replacement + text[index + 1:]
            if candidate == text:
                continue
            value = _ascii_digits(candidate)
            supported = bool(page_values and value and value in page_values)
            candidates.append((0 if supported else 1, order, index, {
                "text": candidate,
                "source": SOURCE,
                "why": f"digit {text[index]} vs {replacement} (measured confusion)",
            }))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    out: List[Dict] = []
    seen = set()
    for _tier, _order, _index, chip in candidates:
        if chip["text"] in seen:
            continue
        seen.add(chip["text"])
        out.append(chip)
        if len(out) >= limit:
            break
    return out


def add_confusions(tokens, limit: int = 2, max_tokens: int = 10) -> int:
    """Append confusion chips to the riskiest queued digit tokens.

    Suspects are flagged digit tokens in queue order (riskiest first), capped
    at `max_tokens` per page so a digit-heavy page cannot flood the review UI.
    Candidates whose value already occurs elsewhere on the page rank first.
    Returns the number of tokens that gained at least one chip.
    """
    from veriscript.document.ocr import review_queue

    page_values = set()
    for token in tokens:
        if not token.has_digits:
            continue
        for text in (token.text, token.alt_text):
            value = _ascii_digits(text or "")
            if value:
                page_values.add(value)

    touched = 0
    for tok in review_queue(tokens):
        if not tok.has_digits:
            continue
        if max_tokens <= 0:
            break
        max_tokens -= 1
        chips = generate(tok.text, limit=limit, page_values=page_values)
        if not chips:
            continue
        existing = {s.get("text") for s in tok.suggestions}
        fresh = [c for c in chips if c["text"] not in existing]
        if fresh:
            tok.suggestions.extend(fresh)
            touched += 1
    return touched
