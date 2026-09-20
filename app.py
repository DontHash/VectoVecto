"""
app.py — Interactive Split-Screen Super-Resolution Studio.
VectorScaling Phase 7: Unified Autonomous Smart Router.

Powered by Gradio 6.20.0 and native hardware-accelerated gr.ImageSlider.
Allows dragging a vertical slider to compare before and after in real time,
visualizing semantic routing maps, and downloading upscaled PNGs and vector SVGs.
"""

import os
import sys
import time
import tempfile
from typing import Optional, Tuple, Any, List, Union
import cv2
import numpy as np
import gradio as gr

# Ensure local VectorScaling workspace is in python path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from smart_upscaler import SmartUpscaler

# Output storage directory
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "web_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Engine cache: one SmartUpscaler per model spec (lazy-loaded on first use)
_ENGINE_CACHE: dict = {}


def _get_upscaler(model_spec: str) -> SmartUpscaler:
    if model_spec not in _ENGINE_CACHE:
        print(f"Initializing VectorScaling engine: {model_spec}")
        _ENGINE_CACHE[model_spec] = SmartUpscaler(model_spec=model_spec)
    return _ENGINE_CACHE[model_spec]


def _compute_device() -> str:
    try:
        import torch
        return "CUDA" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "CPU"


MODEL_CHOICES = [
    ("Auto — best available (fine-tuned -> x4plus)", "auto"),
    ("Real-ESRGAN x4plus (photos, best quality)", "x4plus"),
    ("Real-ESRGAN general v3 (fast)", "x4v3"),
    ("4x-UltraSharp (graphics / anime)", "ncnn:ultrasharp-4x"),
    ("Remacri (photos)", "ncnn:remacri-4x"),
    ("High Fidelity (photos)", "ncnn:high-fidelity-4x"),
]


def process_image(
    input_img: Optional[np.ndarray],
    scale: int,
    mode: str,
    model_choice: str = "auto",
    fast_mode: bool = True,
    grain_val: float = 0.0,
    export_svg_flag: bool = True,
    show_mask_flag: bool = True
):
    """
    Main inference handler for Gradio web interface.
    Returns:
      (slider_tuple, mask_img, png_path, svg_path, status_markdown)
    """
    if input_img is None:
        return None, None, None, None, "Please upload or select an image to upscale."

    t0 = time.time()
    h, w = input_img.shape[:2]
    out_h, out_w = h * scale, w * scale

    # Gradio provides RGB numpy array; convert to BGR for internal OpenCV processing
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    timestamp = int(time.time() * 1000)
    png_path = os.path.join(OUTPUT_DIR, f"upscaled_{timestamp}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"vector_{timestamp}.svg") if export_svg_flag else None
    mask_path = os.path.join(OUTPUT_DIR, f"mask_{timestamp}.png") if show_mask_flag else None

    # Step 1: Run Smart Router Pipeline
    upscaler = _get_upscaler(model_choice)
    out_bgr = upscaler.upscale(
        img=img_bgr,
        scale=scale,
        mode=mode,
        fast=fast_mode,
        grain_strength=grain_val,
        export_svg_path=svg_path,
        export_mask_path=mask_path
    )

    # Convert results back to RGB for Gradio display
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(png_path, out_bgr)

    # Step 2: Create matching baseline for left pane of ImageSlider
    # Scale original LR image with bicubic to exact HR dimensions for 1:1 pixel alignment
    lr_aligned = cv2.resize(input_img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # Slider value: (left_image, right_image) -> (Bicubic Baseline, Smart Upscaled)
    slider_data = (lr_aligned, out_rgb)

    # Step 3: Diagnostic Mask handling
    diag_rgb = None
    if show_mask_flag and mask_path and os.path.exists(mask_path):
        diag_bgr = cv2.imread(mask_path)
        if diag_bgr is not None:
            diag_rgb = cv2.cvtColor(diag_bgr, cv2.COLOR_BGR2RGB)

    # Ensure SVG file actually exists before returning to file component
    actual_svg = svg_path if (svg_path and os.path.exists(svg_path)) else None

    elapsed = time.time() - t0
    engine_label = getattr(upscaler, "engine", None)
    engine_name = engine_label.name if engine_label is not None else ("fidelity" if mode == "fidelity" else "auto")

    status_md = f"""### Processing Complete ({elapsed:.2f}s)
- **Input Dimensions**: `{w} × {h}` px
- **Output Dimensions**: `{out_w} × {out_h}` px (**{scale}× Super-Resolution**)
- **Engine**: `{engine_name}` | **Mode**: `{mode.upper()}` | **Profile**: `{'Fast' if fast_mode else '8-way TTA (quality)'}`
- **Micro-Grain**: `{grain_val:.3f}` | **Device**: `{_compute_device()}`
- **Interactive Slider**: Drag the vertical center divider to compare **Bicubic {scale}x** (left) vs **Upscaled** (right).
"""
    return slider_data, diag_rgb, png_path, actual_svg, status_md


CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: auto;
}
.header-box {
    text-align: center;
    padding: 18px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.header-box h1 {
    font-size: 2.2rem;
    margin-bottom: 4px;
    color: #f8fafc;
    font-weight: 700;
}
.header-box p {
    color: #94a3b8;
    font-size: 1.05rem;
}
"""

def create_app():
    with gr.Blocks(title="VectorScaling — Interactive Studio") as demo:
        with gr.Column(elem_classes=["header-box"]):
                gr.Markdown(
                    """
                    # VectorScaling: Interactive Super-Resolution Studio
                    ### Professional Neural Engine (Real-ESRGAN-class) • Bézier Vector Typography + Faithful Fidelity Mode
                    """
                )

        with gr.Row():
            # Left Column: Controls & Input
            with gr.Column(scale=4):
                gr.Markdown("#### 1. Input Image")
                input_image = gr.Image(
                    label="Upload Image (or click example below)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=280
                )

                gr.Markdown("#### 2. Upscaling Settings")
                with gr.Row():
                    scale_radio = gr.Radio(
                        choices=[2, 4],
                        value=4,
                        label="Scale Factor",
                        info="2x or 4x resolution expansion"
                    )
                    mode_dropdown = gr.Dropdown(
                        choices=[("Auto (Smart Router - Recommended)", "auto"),
                                 ("Natural Photo (pure neural)", "photo"),
                                 ("Fidelity (TV-free unfolding, no hallucination)", "fidelity"),
                                 ("Vector Hybrid (High-Contrast Graphics)", "vector")],
                        value="auto",
                        label="Routing Mode"
                    )

                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="auto",
                    label="Upscaling Engine",
                    info="Auto uses the fine-tuned model when available, else x4plus"
                )

                with gr.Row():
                    fast_toggle = gr.Checkbox(
                        value=True,
                        label="Fast Mode",
                        info="Uncheck for 8-way TTA quality profile (slower)"
                    )
                    grain_slider = gr.Slider(
                        minimum=0.0,
                        maximum=0.05,
                        value=0.0,
                        step=0.002,
                        label="Organic Film Micro-Grain (opt-in)",
                        info="0 = off (recommended). Adds film grain if you want a photographic finish"
                    )

                with gr.Row():
                    svg_checkbox = gr.Checkbox(
                        value=True,
                        label="Export Vector SVG",
                        info="Infinite mathematical Bézier curves"
                    )
                    mask_checkbox = gr.Checkbox(
                        value=True,
                        label="Show Semantic Map",
                        info="Green=Vector, Cyan=Skin, Magenta=Texture"
                    )

                submit_btn = gr.Button("Upscale Image", variant="primary", size="lg")

                gr.Markdown("#### 3. Pre-Loaded Examples")
                example_files = []
                for sample_name in ["De1.jpg", "PrakashJI.jpg"]:
                    sample_p = os.path.join(WORKSPACE_DIR, sample_name)
                    if os.path.exists(sample_p):
                        example_files.append(sample_p)

                if example_files:
                    gr.Examples(
                        examples=example_files,
                        inputs=input_image,
                        label="Click any test case to load into the studio:"
                    )

            # Right Column: Interactive Split-Screen Results & Downloads
            with gr.Column(scale=6):
                gr.Markdown("#### 4. Interactive Split-Screen Inspection")
                result_slider = gr.ImageSlider(
                    label="Drag Center Divider: Bicubic 4x (Left) vs. VectorScaling Smart Output (Right)",
                    type="numpy",
                    show_label=True,
                    height=520,
                    slider_position=50
                )

                status_output = gr.Markdown(
                    "Upload an image and click **Upscale Image** to start interactive inspection."
                )

                with gr.Row():
                    diag_map_display = gr.Image(
                        label="Semantic Diagnostic Map (Routing Decisions)",
                        type="numpy",
                        visible=True,
                        height=240
                    )
                    with gr.Column():
                        download_png = gr.File(label="Download High-Res PNG")
                        download_svg = gr.File(label="Download Resolution-Independent SVG")

        # Wire the event
        submit_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                scale_radio,
                mode_dropdown,
                model_dropdown,
                fast_toggle,
                grain_slider,
                svg_checkbox,
                mask_checkbox
            ],
            outputs=[
                result_slider,
                diag_map_display,
                download_png,
                download_svg,
                status_output
            ]
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch VectorScaling Interactive Studio")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio tunnel share link")
    args = parser.parse_args()

    app = create_app()
    print(f"\n=======================================================")
    print(f"  VectorScaling Interactive Studio running on:")
    print(f"  Local URL: http://127.0.0.1:{args.port}")
    print(f"=======================================================\n")
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS
    )
