"""
lexicon.py — Devanagari lexicon for the `unknown_word` review flag.

The word list is built by `scripts/fetch_nepali_lexicon.py` from two
license-clean sources (Tesseract `nep/nep.wordlist`, Apache-2.0;
`nepali-brihat-sabdakosh-json`, MIT) and is **not distributed in this
repository** (`data/` is gitignored). Absence is not an error: the flag is
simply not emitted. `VERISCRIPT_LEXICON` (legacy `VECTOVECTO_LEXICON` still
accepted) overrides the path.

No heavy imports; safe for the shipped path.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from typing import Collection, Dict, List, Optional

from veriscript import branding

from veriscript.paths import ROOT

BASE_DIR = ROOT
ENV_VAR = "VERISCRIPT_LEXICON"
DEFAULT_LEXICON = os.path.join(BASE_DIR, "data", "lexicon",
                               "nepali_lexicon_v1.txt")
MIN_WORD_LEN = 2

# U+0900-U+0963 (letters + signs), U+0970 (abbreviation sign), U+0971-U+097F.
# Danda/double danda (0964/0965) and Devanagari digits (0966-096F) are not
# word characters; ZWJ/ZWNJ are stripped before matching.
_DEVA_WORD_RE = re.compile(r"[\u0900-\u0963\u0970-\u097f]+")
_DEVA_FULL_RE = re.compile(r"^[\u0900-\u0963\u0970-\u097f]+$")
_ZWJ = "\u200c\u200d"
_CACHE: Dict[str, Optional[Dict]] = {}


def normalize_word(word: str) -> str:
    """NFC + ZWJ/ZWNJ removal; the fetch script and the runtime share this."""
    w = unicodedata.normalize("NFC", word)
    for ch in _ZWJ:
        w = w.replace(ch, "")
    return w.strip()


def is_deva_word(word: str) -> bool:
    """True for a lexicon-eligible word: Devanagari only, >= MIN_WORD_LEN."""
    return len(word) >= MIN_WORD_LEN and bool(_DEVA_FULL_RE.match(word))


def deva_words(text: str) -> List[str]:
    """Devanagari words inside one OCR token (digits/punctuation dropped)."""
    out: List[str] = []
    for raw in _DEVA_WORD_RE.findall(normalize_word(text)):
        if is_deva_word(raw):
            out.append(raw)
    return out


def unknown_words(text: str, lexicon: Optional[Collection[str]]) -> List[str]:
    """Out-of-lexicon Devanagari words in `text` ([] when no lexicon)."""
    if not lexicon:
        return []
    return [w for w in deva_words(text) if w not in lexicon]


def resolve_path(path: Optional[str] = None) -> str:
    """First existing lexicon file (explicit/env, repo copy, data-files)."""
    if path:
        return os.path.abspath(path)
    env = branding.env("LEXICON")
    if env:
        return os.path.abspath(env)
    installed = os.path.join(sys.prefix, "lexicon",
                             os.path.basename(DEFAULT_LEXICON))
    if os.path.isfile(installed):
        return installed
    return DEFAULT_LEXICON


def load_lexicon(path: Optional[str] = None) -> Optional[Dict]:
    """Load and cache the word list; None when missing or malformed."""
    key = resolve_path(path)
    if key in _CACHE:
        return _CACHE[key]
    data: Optional[Dict] = None
    try:
        words = set()
        with open(key, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                w = normalize_word(line)
                if is_deva_word(w):
                    words.add(w)
        if words:
            data = {"words": frozenset(words), "count": len(words),
                    "source": os.path.basename(key), "path": key}
    except Exception:  # noqa: BLE001 - a missing lexicon must never fail OCR
        data = None
    _CACHE[key] = data
    return data
