"""
app.py — VectorScaling Studio: Document Restore (default) + Photo upscaling.

Document tab (the product): drop a scan / phone photo / PDF page, get a cleaned
page, a searchable PDF, an overlay of low-confidence tokens and digit conflicts,
and a transcript. Local only — no network.

Photo tab: the previous Real-ESRGAN-class upscaling studio (Advanced), unchanged
API (`process_image`) for the existing tests.

Powered by Gradio 6 and gr.ImageSlider.
"""

import os
import sys
import time
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

OCR_CHOICES = [
    ("Auto (best available)", "auto"),
    ("RapidOCR (PP-OCRv6, recommended)", "rapidocr"),
    ("Tesseract 5 (optional install)", "tesseract"),
]


# ---------------------------------------------------------------------------
# Document mode
# ---------------------------------------------------------------------------

def _transcript_md(result) -> str:
    flagged = [t for t in result.ocr.tokens if t.flags]
    parts = []
    if flagged:
        parts.append("### Review these (nothing was auto-picked)")
        for t in flagged[:15]:
            alt = f" — alternative reading: `{t.alt_text}`" if t.alt_text else ""
            parts.append(f"- `{t.text}` ({', '.join(t.flags)}){alt}")
    else:
        parts.append("No low-confidence tokens or digit conflicts detected.")
    text = "\n".join(t.text for t in result.ocr.tokens if t.text)
    if text:
        parts.append("### Transcript\n```\n" + text[:4000] + "\n```")
    return "\n".join(parts)


def process_document(
    input_img: Optional[np.ndarray],
    pdf_file: Optional[str],
    ocr_backend: str = "auto",
    deskew_flag: bool = False,
    want_overlay: bool = True,
    want_pdf: bool = True,
    want_txt: bool = True,
):
    """Gradio handler for the document tab.

    Returns: (slider_tuple, overlay_rgb, transcript_md, pdf_path, txt_path, status_md)
    """
    from document_pipeline import run_document_pipeline

    if pdf_file is None and input_img is None:
        return None, None, None, None, None, "Upload an image or a PDF page to begin."

    backend = None if ocr_backend in (None, "auto") else ocr_backend
    ts = int(time.time() * 1000)

    if pdf_file:
        import doc_data
        pages = list(doc_data.pdf_to_pages(pdf_file, dpi=200))[:1]
        if not pages:
            return None, None, None, None, None, "Could not read that PDF."
        _idx, img_bgr, _gt = pages[0]
        stem = f"pdf_{ts}"
    else:
        img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        stem = f"doc_{ts}"

    try:
        result = run_document_pipeline(
            img_bgr, backend=backend, deskew=deskew_flag,
            out_dir=OUTPUT_DIR, stem=stem,
            make_pdf=want_pdf, make_overlay=want_overlay,
            make_txt=want_txt, make_json=True)
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, None, f"**Document pipeline failed:** {e}"

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display_rgb = cv2.cvtColor(result.display_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = display_rgb
    if "overlay" in result.outputs:
        ov = cv2.imread(result.outputs["overlay"])
        if ov is not None:
            overlay_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

    resized = bool(result.meta.get("resized"))
    status = f"""### Restore complete ({result.meta['seconds']:.1f}s)
- **Engine**: `document` (classical restore + {result.meta['backend']}) | **Primary stream**: `{result.meta['primary_stream']}`
- **Deskew applied**: `{result.meta['skew_angle']:.2f}°` | **Device**: `{_compute_device()}`
- **Audit**: {result.status_line}
- **Files**: {' · '.join(os.path.basename(p) for p in result.outputs.values()) or 'none'}
"""
    if resized:
        status += "- **Note**: page was downscaled to 2500 px for processing.\n"

    return ((orig_rgb, display_rgb), overlay_rgb, _transcript_md(result),
            result.outputs.get("pdf"), result.outputs.get("txt"), status)


def _document_example_path() -> Optional[str]:
    """Generate a synthetic invoice example once; return its path."""
    try:
        import doc_data
        path = os.path.join(OUTPUT_DIR, "examples", "synthetic_invoice.png")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img, _gt = doc_data.render_synthetic_invoice(seed=1, dpi=150)
            cv2.imwrite(path, img)
        return path
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Photo mode (unchanged API)
# ---------------------------------------------------------------------------

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
    Main inference handler for the photo interface.
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
    with gr.Blocks(title="VectorScaling — Document Restore Studio") as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # VectorScaling — Document Restore
                ### Local. Searchable. Won't invent the numbers on your bill.
                """
            )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Document tab (the product)
            # ------------------------------------------------------------------
            with gr.Tab("Document (recommended)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        doc_input = gr.Image(
                            label="Page photo / scan (or use the PDF slot below)",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=260,
                        )
                        doc_pdf = gr.File(label="…or a PDF (first page)",
                                          file_types=[".pdf"], height=90)

                        gr.Markdown("#### Settings")
                        doc_ocr = gr.Dropdown(choices=OCR_CHOICES, value="auto",
                                              label="OCR engine")
                        with gr.Row():
                            doc_deskew = gr.Checkbox(value=False, label="Deskew (rotate)",
                                                     info="OCR then runs on the rotated page")
                            doc_overlay = gr.Checkbox(value=True, label="Overlay")
                        with gr.Row():
                            doc_pdf_out = gr.Checkbox(value=True, label="Searchable PDF")
                            doc_txt_out = gr.Checkbox(value=True, label="Transcript .txt")

                        doc_btn = gr.Button("Restore & read", variant="primary", size="lg")

                        example_path = _document_example_path()
                        if example_path:
                            gr.Examples(examples=[example_path], inputs=doc_input,
                                        label="Synthetic invoice example:")

                    with gr.Column(scale=6):
                        doc_slider = gr.ImageSlider(
                            label="Drag divider: original (left) vs restored display (right)",
                            type="numpy", height=500, slider_position=50,
                        )
                        doc_status = gr.Markdown(
                            "Drop a page and press **Restore & read**. "
                            "Nothing leaves this machine."
                        )
                        doc_overlay_img = gr.Image(label="Confidence overlay "
                                                   "(green ok / amber uncertain / red conflict)",
                                                   type="numpy", height=260)
                        doc_transcript = gr.Markdown()
                        with gr.Row():
                            doc_pdf_file = gr.File(label="Searchable PDF")
                            doc_txt_file = gr.File(label="Transcript")

                doc_btn.click(
                    fn=process_document,
                    inputs=[doc_input, doc_pdf, doc_ocr, doc_deskew,
                            doc_overlay, doc_pdf_out, doc_txt_out],
                    outputs=[doc_slider, doc_overlay_img, doc_transcript,
                             doc_pdf_file, doc_txt_file, doc_status],
                )

            # ------------------------------------------------------------------
            # Photo tab (previously the whole studio)
            # ------------------------------------------------------------------
            with gr.Tab("Photo (Advanced)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("#### Input Image")
                        input_image = gr.Image(
                            label="Upload Image (or click example below)",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=280,
                        )

                        gr.Markdown("#### Upscaling Settings")
                        with gr.Row():
                            scale_radio = gr.Radio(choices=[2, 4], value=4,
                                                   label="Scale Factor",
                                                   info="2x or 4x resolution expansion")
                            mode_dropdown = gr.Dropdown(
                                choices=[("Auto (Smart Router - Recommended)", "auto"),
                                         ("Natural Photo (pure neural)", "photo"),
                                         ("Fidelity (TV-free unfolding, no hallucination)", "fidelity"),
                                         ("Vector Hybrid (High-Contrast Graphics)", "vector")],
                                value="auto", label="Routing Mode")

                        model_dropdown = gr.Dropdown(
                            choices=MODEL_CHOICES, value="auto", label="Upscaling Engine",
                            info="Auto uses the fine-tuned model when available, else x4plus")

                        with gr.Row():
                            fast_toggle = gr.Checkbox(value=True, label="Fast Mode",
                                                      info="Uncheck for 8-way TTA (slower)")
                            grain_slider = gr.Slider(
                                minimum=0.0, maximum=0.05, value=0.0, step=0.002,
                                label="Organic Film Micro-Grain (opt-in)",
                                info="0 = off (recommended)")

                        with gr.Row():
                            svg_checkbox = gr.Checkbox(value=True, label="Export Vector SVG",
                                                       info="Infinite mathematical Bézier curves")
                            mask_checkbox = gr.Checkbox(value=True, label="Show Semantic Map",
                                                        info="Green=Vector, Cyan=Skin, Magenta=Texture")

                        submit_btn = gr.Button("Upscale Image", variant="primary", size="lg")

                        example_files = []
                        for sample_name in ["De1.jpg", "PrakashJI.jpg"]:
                            sample_p = os.path.join(WORKSPACE_DIR, sample_name)
                            if os.path.exists(sample_p):
                                example_files.append(sample_p)
                        if example_files:
                            gr.Examples(examples=example_files, inputs=input_image,
                                        label="Photo examples:")

                    with gr.Column(scale=6):
                        result_slider = gr.ImageSlider(
                            label="Drag divider: Bicubic (left) vs upscaled (right)",
                            type="numpy", height=520, slider_position=50)
                        status_output = gr.Markdown(
                            "Upload an image and click **Upscale Image**.")
                        with gr.Row():
                            diag_map_display = gr.Image(
                                label="Semantic Diagnostic Map", type="numpy",
                                visible=True, height=240)
                            with gr.Column():
                                download_png = gr.File(label="Download High-Res PNG")
                                download_svg = gr.File(label="Download SVG")

                submit_btn.click(
                    fn=process_image,
                    inputs=[input_image, scale_radio, mode_dropdown, model_dropdown,
                            fast_toggle, grain_slider, svg_checkbox, mask_checkbox],
                    outputs=[result_slider, diag_map_display, download_png,
                             download_svg, status_output],
                )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch VectorScaling Studio")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio tunnel share link")
    args = parser.parse_args()

    app = create_app()
    print(f"\n=======================================================")
    print(f"  VectorScaling — Document Restore Studio running on:")
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
