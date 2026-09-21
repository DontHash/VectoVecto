"""
document_verifier.py — optional second-model digit verifier (opt-in).

The Devanagari bake-off (Appendix L) found exactly one usable candidate:
bodhan-ai/indic-ocr as a *digit verifier*. When its digit reading disagrees
with the primary engine's, the primary token is wrong 81% of the time and
71% of the primary digit errors are caught (frozen letterpress set).

Contract: flag-only. The verifier never replaces text; a disagreement raises
`cross_model_conflict` on the token and stores the alternative reading in
`alt_text`. The review queue ranks it like a dual-stream conflict.

Requirements when enabled: `hf auth login` with the Indic Open Model License
accepted (the repo is gated), ~1.9 GB of weights (downloaded once), torch +
transformers (already used by the photo upscaler stack). Latency: ~0.44 s per
token on the RTX 2050 - verify suspect tokens only.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

REPO_ID = "bodhan-ai/indic-ocr"

_LICENSE_NOTE = (
    "bodhan-ai/indic-ocr is under the Indic Open Model License 1.0: "
    "self-hosting incl. commercial use is permitted; third-party hosted "
    "access needs written approval; attribution required when redistributed.")


def shim_broken_torchaudio() -> None:
    """Neutralize a broken `torchaudio` install (WinError 127).

    transformers gates `import torchaudio` on package metadata, so a native
    crash fires while importing the vision classes this verifier needs. If the
    real import fails, register a stub module (audio is never used here).
    """
    import importlib.machinery
    import types

    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401
        return
    except ImportError:
        return
    except Exception:  # noqa: BLE001
        pass
    mod = types.ModuleType("torchaudio")
    mod.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    mod.__version__ = "0.0.0-stub"
    mod.__path__ = []
    sys.modules["torchaudio"] = mod


class BodhanDigitVerifier:
    """Batch digit re-reader on the bodhan IndicOCR recognizer (HF path)."""

    name = "bodhan"

    def __init__(self, device: str = "auto", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        self._rec = None
        self._repo = None

    def _ensure(self) -> None:
        if self._rec is not None:
            return
        shim_broken_torchaudio()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(f"digit verifier needs huggingface_hub: {e}") from e
        try:
            self._repo = snapshot_download(REPO_ID)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "digit verifier: cannot download bodhan-ai/indic-ocr. "
                "Run `hf auth login` and accept the model license once. "
                f"({str(e)[:160]})") from e
        sys.path.insert(0, self._repo)
        try:
            import torch  # noqa: F401
            from idp_recognizer import HfRecognizer
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"digit verifier: repo import failed ({e})") from e
        import torch as _torch
        device = self.device
        if device == "auto":
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._rec = HfRecognizer(
            ckpt=os.path.join(self._repo, "weights", "ocr"),
            device=device, batch_size=self.batch_size)
        # Cap generation: the vendored default (2048) lets one junk crop run
        # away - measured 32 crops at 6.0 s/crop uncapped vs 0.82 s/crop with
        # a 64-token cap (W0.2). Digit crops need <16 tokens.
        from idp_types import RecognizerConfig  # vendored repo, on sys.path
        self._rec.config = self._rec.config.merged(max_tokens=64)

    def __call__(self, crops: List[np.ndarray]) -> List[str]:
        self._ensure()
        import cv2
        from PIL import Image
        from idp_contract import prompt_for
        from idp_recognizer import CropRequest
        prompt = prompt_for("Text")
        requests = [CropRequest(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)),
                                prompt) for c in crops]
        return [str(t) for t in self._rec.transcribe(requests)]

    def close(self) -> None:
        self._rec = None


_VERIFIERS = {"bodhan": BodhanDigitVerifier}


def get_digit_verifier(name: Optional[str]):
    """Factory; None/off -> None. Raises with a clear message otherwise."""
    if not name or name == "off":
        return None
    cls = _VERIFIERS.get(name)
    if cls is None:
        raise ValueError(f"unknown digit verifier {name!r}; have {sorted(_VERIFIERS)}")
    return cls()
