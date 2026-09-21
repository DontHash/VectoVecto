"""
train.py — CRNN+CTC trainer for Devanagari lines.

Runs locally (CPU, tiny data) and on Vertex AI (T4, prebuilt PyTorch
container). Data is an npz produced by `deva_crnn.data.export_npz` or by
`scripts/export_training_data.py`.

Usage:
    python -m deva_crnn.train --data data.npz --out out/deva_crnn \
        --epochs 30 --batch 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .charset import build_charset, decode, encode
from .data import IN_H, IN_W, load_npz
from .model import CRNN


class LineDataset(Dataset):
    def __init__(self, images: np.ndarray, texts: List[str], charset: List[str]):
        self.images = images
        self.texts = texts
        self.charset = charset

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        img = torch.from_numpy(self.images[i].astype(np.float32) / 255.0)
        img = (img - 0.5) / 0.5
        target = torch.tensor(encode(self.texts[i], self.charset),
                              dtype=torch.long)
        return img.unsqueeze(0), target, len(target)


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lens = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, targets, lens


def evaluate(model: CRNN, loader: DataLoader, charset: List[str],
             device: torch.device, max_batches: int = 0) -> Dict:
    model.eval()
    exact = total = 0
    with torch.no_grad():
        for i, (imgs, _targets, _lens) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            logits = model(imgs.to(device))          # T, B, C
            preds = logits.argmax(-1).permute(1, 0)  # B, T
            for b, p in enumerate(preds):
                hyp = decode(p.tolist(), charset)
                gt = loader.dataset.texts[total]
                exact += int(hyp == gt)
                total += 1
    return {"lines": total, "exact_match": (exact / total) if total else 0.0}


def train(data_path: str, out_dir: str, epochs: int = 30, batch: int = 64,
          lr: float = 1e-3, val_split: float = 0.05, seed: int = 1,
          max_hours: float = 3.0) -> Dict:
    images, texts = load_npz(data_path)
    charset = build_charset(texts)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    n_val = max(1, int(len(texts) * val_split))
    val_idx, train_idx = order[:n_val], order[n_val:]
    ds_tr = LineDataset(images[train_idx], [texts[i] for i in train_idx], charset)
    ds_va = LineDataset(images[val_idx], [texts[i] for i in val_idx], charset)
    # Worker processes cost ~3 s per epoch on Windows regardless of dataset
    # size; only worth it for real datasets.
    workers = 0 if len(ds_tr) < 512 else 2
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, collate_fn=_collate,
                       num_workers=workers,
                       drop_last=len(ds_tr) >= batch)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False,
                       collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(n_classes=len(charset)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(out_dir, exist_ok=True)
    history: List[Dict] = []
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, targets, lens in dl_tr:
            logits = model(imgs.to(device))          # T, B, C
            T = logits.shape[0]
            input_lens = torch.full((imgs.shape[0],), T, dtype=torch.long)
            loss = ctc(logits, targets, input_lens, lens)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue  # never let one bad batch poison the weights
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
        val = evaluate(model, dl_va, charset, device)
        rec = {"epoch": epoch + 1,
               "loss": float(np.mean(losses)) if losses else None, **val,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch + 1}/{epochs} loss {rec['loss']:.3f} "
              f"val_exact {val['exact_match']:.3f} ({rec['seconds']}s)",
              flush=True)
        torch.save({"model": model.state_dict(), "charset": charset,
                    "in_h": IN_H, "in_w": IN_W, "epoch": epoch + 1},
                   os.path.join(out_dir, "ckpt.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"history": history, "charset_size": len(charset),
                       "lines": len(texts), "device": str(device)}, f, indent=2)
        if (time.time() - t0) / 3600.0 > max_hours:
            print("time budget reached, stopping", flush=True)
            break
    return {"history": history, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description="CRNN+CTC Devanagari line trainer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out/deva_crnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch=args.batch,
          lr=args.lr, seed=args.seed, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
