"""
bakeoff_models.py — dev-only recognizer adapters for the Devanagari bake-off.

Nothing here is part of the shipped path. Each candidate exposes:

    available() -> (bool, reason)
    load(lang) -> context object (models/engines; heavy imports are lazy)
    recognize_lines(ctx, crops, lang) -> List[str]
    recognize_pages(ctx, images, lang) -> List[str]

Candidates and licenses (verified 2026-09-21):
    rapidocr   baseline (Apache-2.0) — PP-OCRv5 devanagari mobile
    tesseract  Apache-2.0, tessdata_best nep/hin (downloaded to data/tessdata)
    trocr      MIT — paudelanil/trocr-devanagari-2 (handwritten Nepali words)
    glmocr     Apache-2.0/MIT — zai-org/GLM-OCR (transformers 5.17)
    bodhan     custom "other" license, gated HF repo — requires `hf auth login`
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402

def _shim_broken_torchaudio() -> None:
    """Neutralize the broken `torchaudio` build on this machine (WinError 127).

    transformers gates `import torchaudio` on `is_torchaudio_available()`,
    which only checks package metadata, so the native crash fires while
    importing vision classes we need. If the real import fails, put a stub in
    sys.modules so transformers skips it (we never use audio).
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
    mod.__path__ = []  # behave like a package for find_spec users
    sys.modules["torchaudio"] = mod


CandidateResult = Tuple[bool, str]

TESSDATA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{name}.traineddata"
TESSDATA_DIR = os.path.join(BASE_DIR, "data", "tessdata")
TESS_LANG_MAP = {None: "eng", "": "eng", "en": "eng", "ne": "nep", "hi": "hin",
                 "nep": "nep", "hin": "hin"}


def _ensure_tessdata(lang_code: str) -> str:
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    path = os.path.join(TESSDATA_DIR, f"{lang_code}.traineddata")
    if not os.path.exists(path):
        url = TESSDATA_URL.format(name=lang_code)
        tmp = path + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "vs-bakeoff/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 16)
        os.replace(tmp, path)
    return path


class Candidate:
    name = "?"
    kind = "both"  # "lines" | "pages" | "both"
    license = "?"

    def available(self) -> CandidateResult:
        return False, "not implemented"

    def load(self, lang: Optional[str] = None):
        raise NotImplementedError

    def recognize_lines(self, ctx, crops: List[np.ndarray],
                        lang: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    def recognize_pages(self, ctx, images: List[np.ndarray],
                        lang: Optional[str] = None) -> List[str]:
        raise NotImplementedError


class RapidOCRCandidate(Candidate):
    name = "rapidocr"
    kind = "both"
    license = "Apache-2.0 (PP-OCRv5 devanagari mobile)"

    def available(self):
        from document_ocr import available_backends
        ok = "rapidocr" in available_backends()
        return ok, "installed" if ok else "rapidocr unavailable"

    def load(self, lang=None):
        from document_ocr import get_backend
        return get_backend("rapidocr")

    def recognize_lines(self, ctx, crops, lang=None):
        return [ctx.recognize_crop(c, lang=lang)[0] for c in crops]

    def recognize_pages(self, ctx, images, lang=None):
        return [ctx.run(img, lang=lang).text for img in images]


class TesseractCandidate(Candidate):
    name = "tesseract-nep"
    kind = "both"
    license = "Apache-2.0 (tessdata_best)"

    @staticmethod
    def _find():
        from document_ocr import TesseractBackend
        return TesseractBackend._find()

    def available(self):
        exe = self._find()
        if not exe:
            return False, "tesseract binary not found"
        try:
            _ensure_tessdata("nep")
            _ensure_tessdata("hin")
        except Exception as e:  # noqa: BLE001
            return False, f"tessdata download failed: {e}"
        return True, exe

    def load(self, lang=None):
        return self._find()

    def _run(self, exe, img, lang, psm):
        code = TESS_LANG_MAP.get(lang, "nep")
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.png")
            doc_data.imwrite_safe(inp, img)
            cmd = [exe, inp, "stdout", "-l", code, "--tessdata-dir", TESSDATA_DIR,
                   "--psm", str(psm)]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"tesseract failed: {proc.stderr[-200:]}")
            return proc.stdout.replace("\r\n", "\n").strip()

    def recognize_lines(self, ctx, crops, lang=None):
        return [self._run(ctx, c, lang, psm=7) for c in crops]

    def recognize_pages(self, ctx, images, lang=None):
        return [self._run(ctx, img, lang, psm=6) for img in images]


class TrOCRCandidate(Candidate):
    name = "trocr"
    kind = "lines"
    license = "MIT (paudelanil/trocr-devanagari-2)"
    model_id = "paudelanil/trocr-devanagari-2"

    def available(self):
        _shim_broken_torchaudio()
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from transformers import TrOCRProcessor  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"missing dep: {e}"
        return True, self.model_id

    def load(self, lang=None):
        _shim_broken_torchaudio()
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained(self.model_id)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(device)
        model.eval()
        return {"processor": processor, "model": model, "device": device,
                "torch": torch}

    def recognize_lines(self, ctx, crops, lang=None):
        from PIL import Image
        import cv2
        torch, processor, model = ctx["torch"], ctx["processor"], ctx["model"]
        out = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pixel_values = processor(images=Image.fromarray(rgb),
                                     return_tensors="pt").pixel_values.to(ctx["device"])
            with torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=128)
            out.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return out


class GLMOCRCandidate(Candidate):
    name = "glmocr"
    kind = "both"
    license = "Apache-2.0 / MIT (zai-org/GLM-OCR)"
    model_id = "zai-org/GLM-OCR"

    def available(self):
        _shim_broken_torchaudio()
        try:
            import torch  # noqa: F401
            import transformers
            if not hasattr(transformers, "AutoModelForImageTextToText"):
                return False, "transformers too old for GLM-OCR"
        except Exception as e:  # noqa: BLE001
            return False, f"missing dep: {e}"
        return True, self.model_id

    def load(self, lang=None):
        _shim_broken_torchaudio()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=device)
        model.eval()
        return {"processor": processor, "model": model, "device": device,
                "torch": torch}

    @staticmethod
    def _prompt(processor, pil):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Transcribe all text in this image exactly as written."},
            ],
        }]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(text=text, images=[pil], return_tensors="pt")

    def _generate(self, ctx, img_bgr):
        from PIL import Image
        import cv2
        torch, processor, model = ctx["torch"], ctx["processor"], ctx["model"]
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._prompt(processor, pil).to(ctx["device"])
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        # drop the prompt echo if the processor includes it
        marker = "exactly as written."
        if marker in text:
            text = text.split(marker, 1)[1]
        return text.strip()

    def recognize_lines(self, ctx, crops, lang=None):
        return [self._generate(ctx, c) for c in crops]

    def recognize_pages(self, ctx, images, lang=None):
        return [self._generate(ctx, img) for img in images]


class BodhanCandidate(Candidate):
    name = "bodhan"
    kind = "both"
    license = "custom 'Bodhan AI Open Model License', gated repo"
    model_id = "bodhan-ai/indic-ocr"

    def available(self):
        try:
            from huggingface_hub import whoami
            whoami()
        except Exception:  # noqa: BLE001
            return False, ("gated: run `hf auth login` first "
                           "(anonymous access is not granted)")
        return True, "logged in; license terms must be reviewed before use"

    def load(self, lang=None):
        from huggingface_hub import snapshot_download
        path = snapshot_download(self.model_id)
        return {"path": path}

    def recognize_lines(self, ctx, crops, lang=None):
        raise NotImplementedError(
            "bodhan entry point: read ARCHITECTURE.md/idp_*.py in the downloaded "
            "repo and wire the inference API (see model card)")

    def recognize_pages(self, ctx, images, lang=None):
        raise NotImplementedError("bodhan entry point not wired yet")


REGISTRY: Dict[str, Candidate] = {
    c.name: c() for c in (RapidOCRCandidate, TesseractCandidate, TrOCRCandidate,
                          GLMOCRCandidate, BodhanCandidate)
}


def list_candidates() -> List[Tuple[str, str, str, bool, str]]:
    out = []
    for name, cand in REGISTRY.items():
        try:
            ok, why = cand.available()
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"error: {e}"
        out.append((name, cand.kind, cand.license, ok, why))
    return out
