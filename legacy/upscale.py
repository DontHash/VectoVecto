"""
upscale.py — Phase 7 Unified Autonomous Super-Resolution CLI.

One-click upscaling that automatically routes text/logos to Bézier vector curves
and photographic skin/textures to Tier-B DRUNet Deep Unfolding.

Examples:
  # Fast auto mode (default, 1-5s on GPU):
  python upscale.py De1.jpg De1_out.png

  # Maximum quality with 8-way TTA + SVG vector export:
  python upscale.py De1.jpg De1_out.png --quality --svg De1_vector.svg

  # Visual inspection of the Smart Router's semantic routing:
  python upscale.py De1.jpg De1_out.png --export-mask De1_mask.png
"""


# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import sys
import argparse
import cv2
import numpy as np

from smart_upscaler import SmartUpscaler, smart_upscale

def parse_args():
    parser = argparse.ArgumentParser(
        description="VectorScaling Phase 7: Unified Autonomous Smart Super-Resolution"
    )
    parser.add_argument("input", help="Path to input image (JPG, PNG, WebP)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Path to output upscaled image (default: <input>_upscaled.png)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4],
                        help="Upscaling factor (default: 4)")
    parser.add_argument("--mode", choices=["auto", "photo", "vector"], default="auto",
                        help="Routing mode: 'auto' (smart router), 'photo' (pure neural), 'vector' (hybrid)")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Fast execution without 8-way TTA (default: True)")
    parser.add_argument("--quality", dest="fast", action="store_false",
                        help="High quality execution with 8-way TTA")
    parser.add_argument("--grain", type=float, default=0.018,
                        help="Organic photographic micro-grain strength (default: 0.018)")
    parser.add_argument("--no-grain", dest="grain", action="store_const", const=0.0,
                        help="Disable organic micro-grain")
    parser.add_argument("--svg", type=str, default=None,
                        help="Optional path to export resolution-independent W3C SVG vector paths")
    parser.add_argument("--export-mask", type=str, default=None,
                        help="Optional path to save semantic routing visualization (green=vector, cyan=skin, magenta=texture)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="Compute device (default: auto-detect CUDA)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image not found at '{args.input}'")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_upscaled.png"

    print("=" * 65)
    print("  VectorScaling — Phase 7 Unified Autonomous Smart Router")
    print("=" * 65)
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Scale       : {args.scale}x")
    print(f"  Mode        : {args.mode}")
    print(f"  Profile     : {'Fast (1-5s)' if args.fast else 'Quality (8-way TTA)'}")
    print(f"  Micro-Grain : {args.grain:.3f}")
    if args.svg:
        print(f"  SVG Export  : {args.svg}")
    if args.export_mask:
        print(f"  Mask Export : {args.export_mask}")
    print("-" * 65)

    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not decode image at '{args.input}'")
        sys.exit(1)

    upscaler = SmartUpscaler(device=args.device)
    out = upscaler.upscale(
        img=img,
        scale=args.scale,
        mode=args.mode,
        fast=args.fast,
        grain_strength=args.grain,
        export_svg_path=args.svg,
        export_mask_path=args.export_mask
    )

    cv2.imwrite(args.output, out)
    print("-" * 65)
    print(f"SUCCESS: Upscaled image saved to {os.path.abspath(args.output)}")
    print(f"Dimensions: {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}")
    print("=" * 65)

if __name__ == "__main__":
    main()
