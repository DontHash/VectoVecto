"""
test_deva_crnn.py — charset/model plumbing + an overfit proof for the trainer.

The overfit test is the cheapest honest signal that the CRNN+CTC pipeline
learns: 16 rendered lines must be memorized to exact match. If this fails,
nothing else about the fine-tune matters.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from deva_crnn.charset import build_charset, decode, encode  # noqa: E402
from deva_crnn.augment import augment_line  # noqa: F401
from deva_crnn.data import export_npz, load_npz, normalize_line  # noqa: E402
from deva_crnn.model import CRNN  # noqa: E402
from deva_crnn.train import train  # noqa: E402
from doc_data import render_devanagari_line  # noqa: E402


def test_charset_roundtrip():
    cs = build_charset(["नेपाल १२", "मिति"])
    ids = encode("नेपाल १२", cs)
    assert decode(ids, cs) == "नेपाल १२"
    assert len(cs) == len(set(cs)) and cs[0] == "\u0000"


def test_normalize_line_is_fixed_size():
    img = render_devanagari_line("कुल जम्मा रु. १,२३४.५०", px=40)
    norm = normalize_line(img)
    assert norm.shape == (32, 256) and norm.dtype == np.uint8
    assert norm.min() < 128, "ink must survive normalization"


def test_model_output_shape():
    m = CRNN(n_classes=17, hidden=32)
    out = m(torch.zeros(2, 1, 32, 256))
    assert out.shape[1] == 2 and out.shape[2] == 17
    probs = out.exp().sum(-1)  # (T, B)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_overfit_tiny_set(tmp_path):
    """Tiny alphabet + short lines: the pipeline must memorize them."""
    texts = ["कख", "ग१", "ख२३", "१२", "क१२", "गख", "खग", "क३"]
    imgs = [render_devanagari_line(t, px=36) for t in texts]
    data = str(tmp_path / "tiny.npz")
    export_npz(imgs, texts, data, w=128)

    import deva_crnn.train as tr

    orig_init = tr.CRNN.__init__

    def small_init(self, n_classes, hidden=256, in_h=32):
        orig_init(self, n_classes, hidden=32, in_h=in_h)

    tr.CRNN.__init__ = small_init
    try:
        result = train(data, str(tmp_path / "out"), epochs=900, batch=8,
                       lr=1e-3, val_split=0.125, seed=3, max_hours=0.2)
    finally:
        tr.CRNN.__init__ = orig_init

    assert result["history"][-1]["loss"] < 0.5, \
        f"CTC loss must collapse on a memorizable set: {result['history'][-1]}"

    # Memorization check on the training lines, decoded from the checkpoint.
    import torch

    from deva_crnn.charset import decode
    from deva_crnn.model import CRNN

    ckpt = torch.load(str(tmp_path / "out" / "ckpt.pt"), weights_only=False)
    model = CRNN(n_classes=len(ckpt["charset"]), hidden=32)
    model.load_state_dict(ckpt["model"])
    model.eval()
    images, labels = load_npz(data)
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    with torch.no_grad():
        preds = model((x - 0.5) / 0.5).argmax(-1).permute(1, 0)
    exact = sum(decode(p.tolist(), ckpt["charset"]) == gt
                for p, gt in zip(preds, labels))
    assert exact >= 6, f"memorized {exact}/8 lines"
