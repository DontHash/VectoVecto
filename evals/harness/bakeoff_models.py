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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import doc_data  # noqa: E402

def _shim_broken_torchaudio() -> None:
    """Delegate to the runtime shim (document_verifier)."""
    from document_verifier import shim_broken_torchaudio
    shim_broken_torchaudio()


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
        import json as _json

        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (RobertaTokenizerFast, TrOCRProcessor,
                                  ViTImageProcessor, VisionEncoderDecoderModel)
        # transformers 5.x dropped the legacy "ViTFeatureExtractor" type key;
        # rebuild the image processor from the raw params. The repo declares
        # the slow RobertaTokenizer, so use the fast class directly.
        cfg = _json.load(open(hf_hub_download(self.model_id,
                                              "preprocessor_config.json"),
                              encoding="utf-8"))
        for legacy in ("image_processor_type", "processor_class",
                       "feature_extractor_type"):
            cfg.pop(legacy, None)
        image_processor = ViTImageProcessor(**cfg)
        tokenizer = RobertaTokenizerFast.from_pretrained(self.model_id)
        processor = TrOCRProcessor(image_processor=image_processor,
                                   tokenizer=tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    license = ("Indic Open Model License 1.0 (self-host OK incl. commercial; "
               "no third-party hosting; >500M MAU / >$250M rev gate)")
    model_id = "bodhan-ai/indic-ocr"

    def available(self):
        try:
            from huggingface_hub import whoami
            whoami()
        except Exception:  # noqa: BLE001
            return False, ("gated: run `hf auth login` first "
                           "(anonymous access is not granted)")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(self.model_id, allow_patterns=["*.py", "*.json"])
        except Exception as e:  # noqa: BLE001
            return False, f"gated: accept the license on the model page ({str(e)[:80]})"
        return True, "logged in; license accepted (self-host permitted)"

    def load(self, lang=None):
        from huggingface_hub import snapshot_download
        _shim_broken_torchaudio()
        repo = snapshot_download(self.model_id)
        return {"repo": repo}

    # -- lines: recognizer only, via the repo's HfRecognizer quickstart path -----
    def _recognizer(self, ctx):
        if "rec" not in ctx:
            sys.path.insert(0, ctx["repo"])
            import torch
            from idp_recognizer import HfRecognizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ctx["rec"] = HfRecognizer(
                ckpt=os.path.join(ctx["repo"], "weights", "ocr"),
                device=device, batch_size=8)
        return ctx["rec"]

    def recognize_lines(self, ctx, crops, lang=None):
        import cv2
        from PIL import Image
        rec = self._recognizer(ctx)  # inserts the repo path
        from idp_contract import prompt_for
        from idp_recognizer import CropRequest
        prompt = prompt_for("Text")
        requests = [CropRequest(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)),
                                prompt) for c in crops]
        return rec.transcribe(requests)

    # -- pages: full two-stage pipeline (layout + block OCR) --------------------
    def _parser(self, ctx):
        if "parser" not in ctx:
            sys.path.insert(0, ctx["repo"])
            from indic_ocr import IndicOCR
            ctx["parser"] = IndicOCR.from_pretrained(ctx["repo"])
        return ctx["parser"]

    def recognize_pages(self, ctx, images, lang=None):
        parser = self._parser(ctx)
        out = []
        with tempfile.TemporaryDirectory() as td:
            for i, img in enumerate(images):
                path = os.path.join(td, f"page_{i}.png")
                doc_data.imwrite_safe(path, img)
                page = parser(path)
                if isinstance(page, dict):
                    out.append(page.get("markdown", ""))
                else:
                    out.append(str(page))
        return out


class Qwen3VLCandidate(Candidate):
    """Qwen3-VL Instruct as an OCR-VLM candidate (Apache-2.0).

    Why this model (2026-09-25 research): the independent real-Devanagari
    stress-test (arXiv 2606.29213) puts Qwen3-VL-8B first among open models
    (chrF++ 75.2, median CER 0.0, 3.3% catastrophic), while Chandra's weights
    are OpenRAIL and PaddleOCR-VL has no Devanagari evidence. `load_in_4bit`
    uses bitsandbytes (needs compute capability >= 7.5: T4/RTX-20xx+; a V100
    is 7.0 and cannot run NF4). The pre-quantized `*-bnb-4bit` repos carry
    their quantization config, so only `device_map` is passed there.
    """
    name = "qwen3vl"
    kind = "both"
    license = "Apache-2.0"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    load_in_4bit = False
    pre_quantized = False
    # A v2 page carries ~2k characters; 1024 tokens would truncate and inflate
    # the page CER. 4096 covers the longest frozen page with headroom.
    max_new_tokens = 4096

    def available(self):
        _shim_broken_torchaudio()
        try:
            import torch  # noqa: F401
            import transformers
            if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
                return False, "transformers too old for Qwen3-VL"
            if self.load_in_4bit:
                import bitsandbytes  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"missing dep: {e}"
        return True, self.model_id

    def load(self, lang=None):
        _shim_broken_torchaudio()
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        kwargs: Dict = {"device_map": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        elif not self.pre_quantized:
            # fp16, not bf16: a V100 (the GPU we can rent) has no bf16 units.
            kwargs["torch_dtype"] = (torch.float16 if torch.cuda.is_available()
                                     else torch.float32)
            if torch.cuda.is_available():
                # An 8B fp16 (~16.4 GB) barely fits a 16 GB V100, and a plain
                # device_map packs it so tightly that the vision pass over a
                # full page OOMs (8.16 GB activation). Reserve ~4 GB of VRAM
                # for activations and let the remainder live in host RAM.
                total = torch.cuda.get_device_properties(0).total_memory / 2**30
                kwargs["max_memory"] = {0: f"{max(2, int(total) - 4)}GiB",
                                        "cpu": "12GiB"}
        processor = AutoProcessor.from_pretrained(self.model_id)
        try:  # cap page visual tokens (Qwen-VL default is 1280*28*28)
            processor.image_processor.max_pixels = 1280 * 28 * 28
        except Exception:  # noqa: BLE001
            pass
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs)
        model.eval()
        return {"processor": processor, "model": model, "torch": torch}

    _PROMPT = ("Transcribe all text in this image exactly as written, in "
               "reading order. Output only the text, no commentary.")

    # Qwen-VL's native visual budget (1280 * 28 * 28 px).
    max_pixels = 1280 * 28 * 28

    def _fit_pixels(self, img_bgr):
        """Cap the image area at the model's native budget (~1.0 Mpx).

        A born-digital page is ~8.4 Mpx; feeding it natively OOMs the vision
        tower on a 16 GB V100 (8.16 GB activation) and the processor knob for
        this is not stable across transformers versions, so downscale here.
        """
        import cv2
        h, w = img_bgr.shape[:2]
        if h * w <= self.max_pixels:
            return img_bgr
        scale = (self.max_pixels / float(h * w)) ** 0.5
        return cv2.resize(img_bgr,
                          (max(1, int(round(w * scale))),
                           max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)

    def _generate(self, ctx, img_bgr) -> str:
        import cv2
        from PIL import Image
        processor, model, torch = ctx["processor"], ctx["model"], ctx["torch"]
        pil = Image.fromarray(cv2.cvtColor(self._fit_pixels(img_bgr),
                                           cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self._PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def recognize_lines(self, ctx, crops, lang=None):
        return [self._generate(ctx, c) for c in crops]

    def recognize_pages(self, ctx, images, lang=None):
        return [self._generate(ctx, img) for img in images]


class Qwen3VL8B4Bit(Qwen3VLCandidate):
    name = "qwen3vl-8b-4bit"
    model_id = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    pre_quantized = True  # quantization config ships with the repo


class Qwen3VL8B4BitRT(Qwen3VLCandidate):
    """Runtime NF4 quantization of the Apache-2.0 checkpoint.

    The pre-quantized unsloth repos trip bitsandbytes' state loader under
    transformers 5.x (`fix_4bit_weight_quant_state_from_module`), so the
    Kaggle T4 path quantizes fresh fp16 weights instead. Needs CC >= 7.5.
    """
    name = "qwen3vl-8b-4bit-rt"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    load_in_4bit = True


class Qwen3VL4B(Qwen3VLCandidate):
    name = "qwen3vl-4b"
    model_id = "Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VL4B4Bit(Qwen3VLCandidate):
    name = "qwen3vl-4b-4bit"
    model_id = "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit"
    pre_quantized = True


class SuryaCandidate(Candidate):
    """Surya OCR 2 (datalab) — VLM-class full-page OCR.

    Surya 2 dropped pure-torch inference: it talks to a local llama.cpp or
    vLLM server. We use the llama.cpp backend (single binary, no Docker):
      LLAMA_CPP_BINARY=<...>/llama-server.exe  (b11270 Vulkan build here)
      SURYA_INFERENCE_BACKEND=llamacpp, SURYA_INFERENCE_PARALLEL=1
    Weights: datalab-to/surya-ocr-2-gguf (1.2 GB + 195 MB mmproj).
    Licence: code Apache-2.0; weights modified OpenRAIL-M (free for research,
    personal use and startups under $5M) — evaluation only here.
    """
    name = "surya"
    kind = "pages"
    license = ("code Apache-2.0 / weights OpenRAIL-M "
               "(free <$5M: research, personal, startups)")

    def available(self):
        try:
            import surya  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"surya not installed in this interpreter ({str(e)[:60]})"
        if not (os.environ.get("LLAMA_CPP_BINARY") or shutil.which("llama-server")):
            return False, "llama-server not found (set LLAMA_CPP_BINARY)"
        return True, "llama.cpp backend"

    def load(self, lang=None):
        os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
        os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")
        from surya.recognition import RecognitionPredictor
        return {"rec": RecognitionPredictor()}

    def recognize_pages(self, ctx, images, lang=None):
        import html as _html
        import re

        import cv2
        from PIL import Image
        from surya.recognition import clean_block_html

        pils = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for img in images]
        out: List[str] = []
        for res in ctx["rec"](pils, full_page=True):
            blocks = sorted(res.blocks, key=lambda b: b.reading_order)
            parts = []
            for b in blocks:
                if getattr(b, "skipped", False):
                    continue
                raw = clean_block_html(getattr(b, "html", "") or "")
                text = _html.unescape(re.sub(r"<[^>]+>", " ", raw))
                if text.strip():
                    parts.append(text.strip())
            out.append("\n".join(parts))
        return out


class PaddleOCRCandidate(Candidate):
    """PaddleOCR 3.x full pipeline (PP-OCRv5), CPU.

    Configuration note (2026-09-30): the model registry has **no Devanagari
    server recognizer** — only `devanagari_PP-OCRv5_mobile_rec` (and v3
    mobile). The generic `PP-OCRv5_server_det` was aborted after burning
    ~5,900 CPU-seconds on a single 8 MP page without returning (impractical
    on CPU); the mobile detector is the standard CPU configuration and is
    what RapidOCR bundles, so this is also the fair like-for-like arm.
    Environment note: paddlepaddle 3.3.1 hits a PIR/oneDNN executor bug on
    this Windows box (`ConvertPirAttribute2RuntimeAttribute`); the venv pins
    paddlepaddle==3.2.2. Licence Apache-2.0.
    """
    name = "paddleocr"
    kind = "pages"
    license = "Apache-2.0 (PP-OCRv5 mobile det + devanagari v5 mobile rec)"

    DET = "PP-OCRv5_mobile_det"
    REC = "devanagari_PP-OCRv5_mobile_rec"

    def available(self):
        try:
            import paddleocr  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"paddleocr not installed ({str(e)[:60]})"
        return True, f"{self.DET} + {self.REC}"

    def load(self, lang=None):
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="hi", ocr_version="PP-OCRv5",
                        text_detection_model_name=self.DET,
                        text_recognition_model_name=self.REC,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
        return {"ocr": ocr}

    @staticmethod
    def _texts(result) -> List[str]:
        d = getattr(result, "json", None)
        if not isinstance(d, dict):
            return []
        res = d.get("res", d)
        if isinstance(res, list) and res:
            res = res[0]
        if isinstance(res, dict):
            texts = res.get("rec_texts")
            if texts is None:
                texts = d.get("rec_texts")
            if texts:
                return [str(t) for t in texts]
        return []

    def recognize_pages(self, ctx, images, lang=None):
        out: List[str] = []
        with tempfile.TemporaryDirectory() as td:
            for i, img in enumerate(images):
                path = os.path.join(td, f"page_{i}.png")
                doc_data.imwrite_safe(path, img)
                res = ctx["ocr"].predict(path)
                texts: List[str] = []
                for r in res or []:
                    texts.extend(self._texts(r))
                out.append("\n".join(texts))
        return out


REGISTRY: Dict[str, Candidate] = {
    c.name: c() for c in (RapidOCRCandidate, TesseractCandidate, TrOCRCandidate,
                          GLMOCRCandidate, BodhanCandidate, Qwen3VLCandidate,
                          Qwen3VL8B4Bit, Qwen3VL8B4BitRT, Qwen3VL4B,
                          Qwen3VL4B4Bit, SuryaCandidate, PaddleOCRCandidate)
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
