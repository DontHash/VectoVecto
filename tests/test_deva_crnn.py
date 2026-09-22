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
from deva_crnn.predict import beam_search_decode  # noqa: E402
from deva_crnn.train import train  # noqa: E402
from doc_data import render_devanagari_line  # noqa: E402


def _collapse(path, charset):
    out, prev = [], None
    for i in path:
        if i != prev and i != 0:
            out.append(charset[i])
        prev = i
    return "".join(out)


def test_beam_search_matches_greedy_on_confident_paths():
    charset = ["\u0000", "क", "ख", "ग"]
    # T=5, C=4; confidently alternating classes with blanks between repeats
    log_probs = np.full((5, 4), -20.0)
    log_probs[0, 1] = -0.01   # क
    log_probs[1, 0] = -0.01   # blank
    log_probs[2, 1] = -0.01   # क (repeat needs the blank)
    log_probs[3, 0] = -0.01
    log_probs[4, 2] = -0.01   # ख
    assert beam_search_decode(log_probs, charset, beam_width=4) == "ककख"


def test_beam_search_never_loses_to_greedy_brute_force():
    """Brute-force all alignments: beam's prefix must be at least as likely."""
    charset = ["\u0000", "क", "ख"]
    rng = np.random.default_rng(4)
    log_probs = np.log(rng.dirichlet([1.0, 1.0, 1.0], size=4))
    # exact prefix probabilities over all 3^4 paths
    from itertools import product
    probs = {}
    for path in product(range(3), repeat=4):
        p = float(np.exp(sum(log_probs[t, c] for t, c in enumerate(path))))
        probs[_collapse(path, charset)] = probs.get(_collapse(path, charset),
                                                    0.0) + p
    greedy_path = log_probs.argmax(axis=1).tolist()
    greedy_text = _collapse(greedy_path, charset)
    beam_text = beam_search_decode(log_probs, charset, beam_width=8)
    assert probs[beam_text] >= probs[greedy_text] - 1e-12


def test_heavy_augment_is_deterministic_and_keeps_ink():
    img = render_devanagari_line("मिति २०८१-०४-२७", px=40)
    r1 = augment_line(img, np.random.default_rng(5), level="heavy")
    r2 = augment_line(img, np.random.default_rng(5), level="heavy")
    assert r1.shape == img.shape and r1.dtype == np.uint8
    assert np.array_equal(r1, r2), "same rng must give the same image"
    light = augment_line(img, np.random.default_rng(5), level="light")
    assert not np.array_equal(r1, light)
    for seed in (0, 1, 2, 3):
        out = augment_line(img, np.random.default_rng(seed), level="heavy")
        assert int((out.min(axis=2) < 160).sum()) > 20, \
            "heavy augmentation must not wash the ink out"
        assert out.std() > 8, "heavy augmentation must not flatten to one tone"


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


def test_model_accepts_48px_input():
    m = CRNN(n_classes=17, hidden=32, in_h=48)
    out = m(torch.zeros(2, 1, 48, 256))
    assert out.shape[1] == 2 and out.shape[2] == 17


def test_model_accepts_64px_input_and_wider_hidden():
    m = CRNN(n_classes=17, hidden=32, in_h=64)
    out = m(torch.zeros(2, 1, 64, 256))
    assert out.shape[1] == 2 and out.shape[2] == 17
    assert out.shape[0] == 64  # T = W/4 timesteps
    m2 = CRNN(n_classes=17, hidden=64, in_h=48)
    out2 = m2(torch.zeros(1, 1, 48, 512))
    assert out2.shape[0] == 128


def test_48px_round_trip_through_trainer(tmp_path):
    """A 48-px npz trains, checkpoints in_h=48, and predicts via that height."""
    from deva_crnn.predict import load_model, recognize_lines

    texts = ["कख", "ग१", "ख२३", "१२"]
    imgs = [render_devanagari_line(t, px=48) for t in texts]
    data = str(tmp_path / "tiny48.npz")
    export_npz(imgs, texts, data, h=48, w=128)

    import deva_crnn.predict as pr
    import deva_crnn.train as tr
    orig_train_init = tr.CRNN.__init__
    orig_pred_init = pr.CRNN.__init__

    def small_init(self, n_classes, hidden=256, in_h=32):
        orig_train_init(self, n_classes, hidden=32, in_h=in_h)

    tr.CRNN.__init__ = small_init
    pr.CRNN.__init__ = small_init
    try:
        tr.train(data, str(tmp_path / "out48"), epochs=2, batch=4, lr=1e-3,
                 val_split=0.25, seed=2, max_hours=0.1, in_h=48, in_w=128)
        ckpt = torch.load(str(tmp_path / "out48" / "ckpt.pt"),
                          weights_only=False)
        assert ckpt["in_h"] == 48
        model, charset = load_model(str(tmp_path / "out48" / "ckpt.pt"), "cpu")
        assert getattr(model, "in_h") == 48
        out = recognize_lines(model, charset, imgs[:2])
    finally:
        tr.CRNN.__init__ = orig_train_init
        pr.CRNN.__init__ = orig_pred_init
    assert len(out) == 2 and all(isinstance(t, str) for t in out)


def test_cosine_schedule_and_save_best(tmp_path):
    """Fine-tune path: cosine LR decay runs and ckpt_best.pt is written."""
    texts = ["कख", "ग१", "ख२३", "१२", "क१२", "गख"]
    imgs = [render_devanagari_line(t, px=36) for t in texts]
    data = str(tmp_path / "cs.npz")
    export_npz(imgs, texts, data, w=64)

    import deva_crnn.train as tr
    orig_init = tr.CRNN.__init__

    def small_init(self, n_classes, hidden=256, in_h=32):
        orig_init(self, n_classes, hidden=32, in_h=in_h)

    tr.CRNN.__init__ = small_init
    try:
        res = tr.train(data, str(tmp_path / "cs_out"), epochs=3, batch=3,
                       lr=1e-3, val_split=0.33, seed=5, max_hours=0.1,
                       lr_schedule="cosine", save_best=True, in_w=64)
    finally:
        tr.CRNN.__init__ = orig_init
    assert len(res["history"]) == 3
    assert (tmp_path / "cs_out" / "ckpt_best.pt").exists()
    assert (tmp_path / "cs_out" / "ckpt.pt").exists()


def test_width_round_trip_through_trainer(tmp_path):
    """A wide (h=48, w=320) npz trains, stores in_w, and predicts at it."""
    from deva_crnn.predict import load_model, recognize_lines

    texts = ["कख", "ग१", "ख२३", "१२"]
    imgs = [render_devanagari_line(t, px=48) for t in texts]
    data = str(tmp_path / "wide.npz")
    export_npz(imgs, texts, data, h=48, w=320)

    import deva_crnn.predict as pr
    import deva_crnn.train as tr
    orig = tr.CRNN.__init__
    orig_pr = pr.CRNN.__init__

    def small(self, n_classes, hidden=256, in_h=32):
        orig(self, n_classes, hidden=32, in_h=in_h)

    tr.CRNN.__init__ = small
    pr.CRNN.__init__ = small
    try:
        tr.train(data, str(tmp_path / "wide_out"), epochs=2, batch=4,
                 lr=1e-3, val_split=0.25, seed=6, max_hours=0.1, in_h=48,
                 in_w=320)
        ckpt = torch.load(str(tmp_path / "wide_out" / "ckpt.pt"),
                          weights_only=False)
        assert ckpt["in_w"] == 320
        model, charset = load_model(str(tmp_path / "wide_out" / "ckpt.pt"),
                                    "cpu")
        assert getattr(model, "in_w") == 320
        out = recognize_lines(model, charset, imgs[:2])
    finally:
        tr.CRNN.__init__ = orig
        pr.CRNN.__init__ = orig_pr
    assert len(out) == 2


def test_trainer_rejects_height_mismatch(tmp_path):
    texts = ["कख", "ग१"]
    imgs = [render_devanagari_line(t, px=36) for t in texts]
    data = str(tmp_path / "h32.npz")
    export_npz(imgs, texts, data, h=32, w=64)
    import deva_crnn.train as tr
    try:
        tr.train(data, str(tmp_path / "out_mismatch"), epochs=1, batch=2,
                 in_h=48)
    except SystemExit as e:
        assert "re-export" in str(e)
    else:
        raise AssertionError("height mismatch must abort before training")


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
                       lr=1e-3, val_split=0.125, seed=3, max_hours=0.2,
                       in_w=128)
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
