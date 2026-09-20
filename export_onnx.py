"""
export_onnx.py — export a trained/pretrained SR generator to ONNX (fp32/fp16)
for fast local inference on the RTX 2050 (onnxruntime-directml).

Supported inputs:
  * RRDBNet weights/RealESRGAN_x4plus.pth (params_ema)
  * SRVGGNetCompact weights/realesr-general-x4v3.pth
  * train_v2.py checkpoints ({"g": ...} or EMA {"params": ...})

Usage:
  python export_onnx.py --checkpoint artifacts/tier_c/g_ema.pth --out weights/onnx/tier_c_x4_fp16.onnx --fp16
  python export_onnx.py --checkpoint weights/RealESRGAN_x4plus.pth --out weights/onnx/x4plus_fp16.onnx --fp16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rrdbnet import RRDBNet  # noqa: E402
from srvggnet import SRVGGNetCompact  # noqa: E402
from sr_engine import detect_arch, load_state_dict_any, build_generator  # noqa: E402


def load_state(path: str):
    return load_state_dict_any(path)


def build_model(sd, scale: int, weights_path: str) -> torch.nn.Module:
    arch = detect_arch(sd)
    if arch == "unfolding":
        raise SystemExit("deep-unfolding checkpoints are not exported here; "
                         "use the Tier-B engine path instead")
    model, _ = build_generator(weights_path, scale, "cpu")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-h", type=int, default=64)
    ap.add_argument("--check-w", type=int, default=64)
    args = ap.parse_args()

    sd = load_state(args.checkpoint)
    model = build_model(sd, args.scale, args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] arch={detect_arch(sd)} params={n_params/1e6:.3f}M scale={args.scale}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.randn(1, 3, args.check_h, args.check_w)
    tmp_path = args.out + ".fp32.tmp.onnx"
    torch.onnx.export(
        model, dummy, tmp_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h*4", 3: "w*4"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[export] wrote {tmp_path}")

    final_path = tmp_path
    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            onnx_model = onnx.load(tmp_path)
            onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            final_path = args.out
            onnx.save(onnx_model, final_path)
            os.remove(tmp_path)
            print(f"[export] converted to fp16 -> {final_path}")
        except ImportError:
            print("[export] onnxconverter-common not installed; keeping fp32")
            os.replace(tmp_path, args.out)
    else:
        os.replace(tmp_path, args.out)

    # --- numerical check vs pytorch ---
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(final_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            y_torch = model(dummy).numpy()
        y_ort = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.abs(y_torch - y_ort).max())
        print(f"[export] onnx-vs-torch max abs diff: {diff:.6f} "
              f"({'OK' if diff < 1e-2 else 'CHECK!'})")
    except ImportError:
        print("[export] onnxruntime not available for verification")

    print(f"[export] done: {args.out}")


if __name__ == "__main__":
    main()
