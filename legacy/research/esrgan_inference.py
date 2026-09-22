"""
esrgan_inference.py — cleaned, enhanced, professional upscaling via RRDBNet.

Defaults target perceptual quality (the stated goal), not raw throughput:
  - 8-way Test-Time Augmentation ON  (removes directional streaks = "cleaned")
  - whole-image inference            (no seams, no context starvation)
  - optional tiling fallback         (kicks in only when VRAM is insufficient)
  - alpha channel preserved          (transparent PNGs no longer break)
  - fp16 on CUDA                     (zero quality loss, ~2-3x faster -> makes TTA affordable)
  - NO post-processing by default   ("professional" = let the model's output be final)
  - optional --enhance for gentle unsharp only, no color/contrast boost

Backward compatible: omit all args and it uses IMAGE_PATH/OUTPUT_PATH globals below
with a Tk file-dialog fallback, exactly like the previous version.

Recommended weights for photos: weights/RealESRGAN_x4plus.pth  (general-purpose, trained
with second-order degradation: blur -> down -> noise -> JPEG, so it denoises+de-JPEGs
while upscaling). The default weights/4x-UltraSharp.pth is anime/illustration-tuned and
over-sharpens photographic content.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

from rrdbnet import RRDBNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "weights", "4x-UltraSharp.pth")
RECOMMENDED_PHOTO_WEIGHTS = os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth")

IMAGE_PATH = "./idbhuntu.jpg"
OUTPUT_PATH = "./idbhuntu_upscaled_pytorch.png"


# ----------------------------------------------------------------------------
# 1. Weights loading
# ----------------------------------------------------------------------------
def _remap_wang_esrgan_keys(sd):
    """
    Some community ESRGAN weights (e.g. 4x-UltraSharp.pth) use the original
    xinntao RRDBNet-as-nested-Sequential key scheme:
        model.0.*                                  -> conv_first.*
        model.1.sub.<i>.RDB<n>.conv<m>.0.*         -> body.<i>.rdb<n>.conv<m>.*
        model.1.sub.23.*                           -> conv_body.*
        model.3.*  / model.6.* / model.8.* / model.10.* -> conv_up1/up2/hr/last.*
    rrdbnet.RRDBNet uses the BasicSR clean naming (conv_first, body.<i>.rdb<n>.conv<m>, conv_body, ...).
    This remaps the community-form keys to the clean form so we can load on a single architecture.
    """
    out = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("model.0."):
            nk = "conv_first." + k[len("model.0."):]
        elif k.startswith("model.1.sub."):
            rest = k[len("model.1.sub."):]
            idx_dot = rest.find(".")
            sub_idx = rest[:idx_dot]
            tail = rest[idx_dot + 1:]
            if sub_idx == "23":
                nk = "conv_body." + tail
            else:
                # tail like: RDB1.conv1.0.weight -> rdb1.conv1.weight
                m = tail.startswith("RDB")
                if m:
                    rdb_id = tail[3]                                # '1'/'2'/'3'
                    conv_part = tail[5:]                            # e.g. "conv1.0.weight"
                    conv_part = conv_part.replace(".0.", ".", 1) if conv_part.startswith("conv") and ".0." in conv_part else conv_part
                    nk = f"body.{sub_idx}.rdb{rdb_id}.{conv_part}"
        elif k.startswith("model.3."):
            nk = "conv_up1." + k[len("model.3."):]
        elif k.startswith("model.6."):
            nk = "conv_up2." + k[len("model.6."):]
        elif k.startswith("model.8."):
            nk = "conv_hr." + k[len("model.8."):]
        elif k.startswith("model.10."):
            nk = "conv_last." + k[len("model.10."):]
        out[nk] = v
    return out


def load_weights(path, device):
    """Load RRDBNet state dict from .pth. Handles params_ema / params / bare forms,
    and remaps community ESRGAN weights that use the original Wang-style key scheme."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    print(f"Loading weights: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    elif isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]

    # Detect community/Wang-style keys and remap to clean BasicSR naming.
    if any(k.startswith("model.") for k in sd.keys()):
        sd = _remap_wang_esrgan_keys(sd)
        print("  [remap] translated Wang-style ESRGAN keys to BasicSR naming.")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(device)
    return model


# ----------------------------------------------------------------------------
# 2. Preprocessing (cv2 path, with explicit RGBA handling)
# ----------------------------------------------------------------------------
def preprocess(path):
    """
    Returns (rgb_tensor[1,3,H,W] float32 in [0,1] on CPU, alpha_np|None, (W,H)).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    alpha_np = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        alpha_np = img[:, :, 3].copy()
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGRA2BGR)

    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # to RGB
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    h, w = tensor.shape[-2:]
    return tensor, alpha_np, (w, h)


# ----------------------------------------------------------------------------
# 3. Inference primitives
# ----------------------------------------------------------------------------
def _run_one(model, x):
    return model(x)


def tta_forward(model, x):
    """8-way Test-Time Augmentation: 4 rotations x {orig, hflip}. Averages predictions."""
    import sys
    out_sum = None
    n = 0
    for k in range(4):
        rot = torch.rot90(x, k, [2, 3])
        pred = _run_one(model, rot)
        pred = torch.rot90(pred, -k, [2, 3])
        if out_sum is None:
            out_sum = pred
        else:
            out_sum = out_sum + pred
        n += 1
        print(f"    [tta] pass {n}/8 done", flush=True)

        flip = torch.flip(rot, [3])
        pred_f = _run_one(model, flip)
        pred_f = torch.rot90(torch.flip(pred_f, [3]), -k, [2, 3])
        out_sum = out_sum + pred_f
        n += 1
        print(f"    [tta] pass {n}/8 done", flush=True)

    return out_sum / 8.0


@torch.no_grad()
def whole_image_inference(model, x, use_tta):
    return tta_forward(model, x) if use_tta else _run_one(model, x)


def _cosine_ramp(tile_h, tile_w, overlap):
    """2D weight mask: 1 in the center, cosine taper to 0 across the overlap band."""
    def _ramp1(n, ov):
        r = np.ones(n, dtype=np.float32)
        if ov > 0:
            ramp = np.sin(np.linspace(0, np.pi / 2, ov, dtype=np.float32))
            r[:ov] = ramp
            r[-ov:] = ramp[::-1]
        return r
    hy = _ramp1(tile_h, min(overlap, tile_h // 2))
    hx = _ramp1(tile_w, min(overlap, tile_w // 2))
    return torch.from_numpy(np.outer(hy, hx))


@torch.no_grad()
def tiled_inference(model, x, scale, tile, overlap, use_tta):
    """
    Tile the input with overlap, run inference per tile, blend with cosine ramps.
    x: [1,3,H,W]. Returns [1,3,H*scale,W*scale].
    """
    _, _, H, W = x.shape
    # Reflect-pad so the image dims are exact multiples of `tile`.
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = x.shape

    out = torch.zeros((1, 3, Hp * scale, Wp * scale), dtype=x.dtype, device=x.device)
    wsum = torch.zeros((1, 1, Hp * scale, Wp * scale), dtype=x.dtype, device=x.device)

    # We'll process tiles on the ORIGINAL grid (no overlap doubled), pulling a
    # (tile+2*overlap) crop for context but writing only the (tile*scale) center.
    stride = tile
    ctx = overlap
    y = 0
    while y < Hp:
        x_pix = 0
        h = min(tile, Hp - y)
        while x_pix < Wp:
            w = min(tile, Wp - x_pix)

            # context crop (reflect-padded at borders)
            y0, y1 = max(0, y - ctx), min(Hp, y + h + ctx)
            x0, x1 = max(0, x_pix - ctx), min(Wp, x_pix + w + ctx)
            crop = x[:, :, y0:y1, x0:x1]

            pred = whole_image_inference(model, crop, use_tta)  # [1,3,ch*scale,cw*scale]

            # The center of `pred` corresponds to the (y..y+h, x_pix..x_pix+w) region of
            # the input (the context band maps to the outer ring of `pred`).
            cy0 = (y0 != y) * ctx * scale
            cx0 = (x0 != x_pix) * ctx * scale
            cy1 = cy0 + h * scale
            cx1 = cx0 + w * scale
            center = pred[:, :, cy0:cy1, cx0:cx1]

            # Weight mask of size (h*scale, w*scale). Full 1 unless this tile is at the
            # border (border tiles have nothing to blend against on the outer side).
            mask_h = h * scale
            mask_w = w * scale
            weight = _cosine_ramp(mask_h, mask_w, overlap * scale)
            # Zero out the outer-edge taper if this tile touches the image border
            # (no neighbor there to average with -> keep those pixels at full weight).
            if y == 0:
                weight[:overlap * scale, :] = 1.0 if h == tile else weight[:overlap * scale, :]
            if x_pix == 0:
                weight[:, :overlap * scale] = 1.0 if w == tile else weight[:, :overlap * scale]
            if y + h >= Hp:
                weight[-overlap * scale:, :] = 1.0
            if x_pix + w >= Wp:
                weight[:, -overlap * scale:] = 1.0
            weight = weight.unsqueeze(0).unsqueeze(0).to(x.device)

            oy = y * scale
            ox = x_pix * scale
            out[:, :, oy:oy + h * scale, ox:ox + w * scale] += center * weight
            wsum[:, :, oy:oy + h * scale, ox:ox + w * scale] += weight

            x_pix += stride
        y += stride

    out = out / wsum.clamp_min(1e-8)
    # Crop back to original-image dimensions.
    out = out[:, :, :H * scale, :W * scale]
    return out


# ----------------------------------------------------------------------------
# 4. VRAM estimation (auto-tiling decision on CUDA)
# ----------------------------------------------------------------------------
def estimate_vram_bytes(h, w, scale, fp16):
    bytes_per_elem = 2 if fp16 else 4
    # Rough: 3 channels in, 3 out, ~8x activations across the 23 RRDBs.
    return h * w * scale * scale * 3 * bytes_per_elem * 8


def should_tile(h, w, scale, fp16, device):
    if device.type != "cuda":
        # On CPU, paging handles large tensors; tiling slows things down.
        return False, 256, 64
    free, _ = torch.cuda.mem_get_info(device)
    need = estimate_vram_bytes(h, w, scale, fp16)
    if need < 0.8 * free:
        return False, 0, 0
    return True, 256, 64


# ----------------------------------------------------------------------------
# 5. Postprocessing (alpha re-merge + optional gentle enhance)
# ----------------------------------------------------------------------------
def gentle_enhance(bgr_uint8):
    """
    Optional post-processing: very gentle bilateral smoothing + low-intensity
    unsharp mask only. NO color saturation boost, NO contrast boost (those were
    the over-processing culprits that made the old pipeline look unprofessional).
    """
    pil = Image.fromarray(cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB))
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=30, threshold=2))
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=30)
    return bgr


def postprocess(out_tensor, alpha_np, out_w, out_h, do_enhance):
    """
    out_tensor: [1,3,OH,OW] in [0,1] on CPU. Returns uint8 HxWxC for cv2.imwrite.
    Re-merges alpha if present. Applies gentle enhance if requested.
    """
    out = out_tensor.squeeze(0).clamp(0, 1).detach().cpu().float().numpy()
    out = np.transpose(out, (1, 2, 0))           # HWC, RGB
    out = (out * 255.0).round().astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    if do_enhance:
        out = gentle_enhance(out)

    if alpha_np is not None:
        alpha = cv2.resize(alpha_np, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        out = np.dstack([out, alpha])
    return out


def save_image(img_uint8, path):
    # cv2.imwrite can write 4-channel BGRA to PNG; for non-PNG with alpha we force PNG.
    if img_uint8.ndim == 3 and img_uint8.shape[2] == 4:
        if not path.lower().endswith(".png"):
            base, _ = os.path.splitext(path)
            path = base + ".png"
            print(f"  [alpha] output has 4 channels, switching to PNG: {path}")
    ok = cv2.imwrite(path, img_uint8)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    return path


# ----------------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RRDBNet upscaling (cleaned, enhanced, professional).",
        epilog=(f"Recommended weights for photos: {RECOMMENDED_PHOTO_WEIGHTS}\n"
                "Example: python esrgan_inference.py -i De1.jpg -o De1_pro.png "
                f"--weights {RECOMMENDED_PHOTO_WEIGHTS} --tta"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", default=None, help="Input image (omit to use file dialog).")
    p.add_argument("-o", "--output", default=None, help="Output image path.")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help=(f"RRDBNet weights .pth (default: {os.path.basename(DEFAULT_WEIGHTS)}; "
                         f"for photos use {os.path.basename(RECOMMENDED_PHOTO_WEIGHTS)})."))
    p.add_argument("--tta", dest="tta", action="store_true", default=True,
                   help="8-way Test-Time Augmentation ON (default; better quality, 8x slower).")
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="Disable TTA (8x faster, slightly more artifacts).")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None,
                   help="fp16 autocast on CUDA (default ON for CUDA, OFF for CPU).")
    p.add_argument("--fp32", dest="fp16", action="store_false",
                   help="Force fp32 (slower on CUDA; no-op on CPU).")
    p.add_argument("--tile", type=int, default=0,
                   help="Tile size for tiling fallback (0 = auto, whole-image when VRAM allows).")
    p.add_argument("--overlap", type=int, default=64,
                   help="Tile overlap in pixels (used only when --tile > 0).")
    p.add_argument("--enhance", dest="enhance", action="store_true", default=False,
                   help="Apply gentle unsharp+bilateral post-processing (default off; "
                        "the model output is already the final image).")
    p.add_argument("--no-enhance", dest="enhance", action="store_false")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Compute device (default: auto = cuda if available else cpu).")
    return p.parse_args()


def resolve_input_output(args):
    img_path = args.input
    if img_path is None:
        img_path = IMAGE_PATH
    if not os.path.exists(img_path):
        # Tk file-dialog fallback (preserves backward behavior)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            img_path = filedialog.askopenfilename(
                title="Select an image to upscale (4x)",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
            )
            root.destroy()
        except Exception as e:
            print(f"Could not open GUI file dialog. Error: {e}", file=sys.stderr)
            return None, None
        if not img_path:
            print("No file selected. Exiting.", file=sys.stderr)
            return None, None

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(img_path)
        out_path = base + "_upscaled_pytorch.png"
    return img_path, out_path


def main():
    args = parse_args()

    # Backward-compat: if invoked with no CLI args, use the module-level globals
    # + Tk fallback. argparse still runs (defaults populate everything).
    img_path, out_path = resolve_input_output(args)
    if not img_path:
        return

    # Device selection.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # fp16 default: ON for cuda, OFF for cpu. CPU fp16 autocast is a no-op here.
    use_fp16 = args.fp16 if args.fp16 is not None else (device.type == "cuda")
    if use_fp16 and device.type != "cuda":
        print("  [fp16] requested on CPU -> no-op (CPU autocast is fp32). Using fp32.")
        use_fp16 = False

    print(f"Using device: {device}{' (fp16)' if use_fp16 else ''}")
    print(f"TTA: {'ON (8x passes)' if args.tta else 'OFF'}")

    # Load model.
    model = load_weights(args.weights, device)

    # Preprocess.
    print(f"\nReading image: {img_path}")
    rgb_tensor, alpha_np, (W, H) = preprocess(img_path)
    rgb_tensor = rgb_tensor.to(device)
    print(f"Original dimensions: {W}x{H}")
    if alpha_np is not None:
        print("  [alpha] RGBA input detected; alpha will be preserved through inference.")

    # Decide tiling.
    if args.tile > 0:
        do_tile = True
        tile, overlap = args.tile, args.overlap
        print(f"  Tiling forced: tile={tile}, overlap={overlap}")
    else:
        do_tile, tile, overlap = should_tile(H, W, model.scale, use_fp16, device)
        if do_tile:
            print(f"  [auto] image too large for whole-image VRAM; tiling tile={tile}, overlap={overlap}")

    # Optional fp16 autocast wrapper.
    global _run_one
    raw_run = _run_one

    def _fp_run(m, x):
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return m(x).float()
        return m(x)

    _run_one = _fp_run

    # Inference.
    scale = model.scale
    t0 = time.time()
    print("\nRunning neural network... please wait.")
    if do_tile:
        out_tensor = tiled_inference(model, rgb_tensor, scale, tile, overlap, args.tta)
    else:
        out_tensor = whole_image_inference(model, rgb_tensor, args.tta)
    out_tensor = out_tensor.cpu()
    dt = time.time() - t0
    _, _, OH, OW = out_tensor.shape
    print(f"Inference done in {dt:.1f}s -> {OW}x{OH}")

    # Postprocess + save.
    final = postprocess(out_tensor, alpha_np, OW, OH, args.enhance)
    if args.enhance:
        print("  [enhance] applied gentle unsharp + bilateral (no color/contrast boost).")
    out_path = save_image(final, out_path)
    print(f"\nSUCCESS! Upscaled image saved to: {os.path.abspath(out_path)}")
    print(f"New dimensions: {OW}x{OH}")


if __name__ == "__main__":
    main()