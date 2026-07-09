# Math-Based Image Upscaling — Exploratory Roadmap

**Status:** This is a rough, exploratory map, not a rigid spec. Phases can be reordered, merged, cut, or expanded once real numbers come in from the eval harness. Treat every phase as an independently swappable module: `numpy array in, numpy array out`, so any combination can be A/B tested without rewrites.

## The honest math ceiling

Upscaling is an ill-posed inverse problem: `y = D H x + noise`, where `x` is the unknown high-res image, `H` is a blur/anti-alias filter, and `D` is downsampling. Infinitely many `x` map to the same `y`. Classical math (FFT, wavelets, sparse coding, self-similarity) gives principled ways to pick the *most plausible* `x` under some prior — bandlimitedness, sparsity, self-similarity, smoothness. What pure math cannot do is hallucinate genuinely new high-frequency detail with zero trace in the low-res signal — that's what deep generative models buy you over classical methods.

**Realistic target:** meaningfully better than bicubic/bilinear, competitive with pre-deep-learning state of the art (roughly pre-SRCNN era), near-lossless on band-limited/self-similar content (text, fabric, repeating patterns), but visibly softer than learned models on stochastic fine detail (skin pores, hair, foliage). Prioritizing mathematically rigorous techniques (PDEs, TV minimization, Implicit Functions, Fractals) over heuristics.

---

## Phase 0 — Eval harness first

Build this before writing any upscaler. Needed:
- A set of HR source images
- A controlled, known downsampling function (so ground truth is known)
- PSNR / SSIM / LPIPS scoring
- Side-by-side crop viewer for eyeballing artifacts

Every later phase is judged against this harness. Skipping this is the most common way this kind of project stalls — you need numbers, not vibes, to know if phase N+1 actually beat phase N.

## Phase 1 — Baselines

- Proper bicubic and Lanczos resampling (correct anti-alias pre-filtering on downsampling).
- Naive FFT zero-padding upscaling: DFT the LR image, zero-pad in frequency, inverse transform. This is literally "ideal bandlimited interpolation" — the clean textbook link between FFT and upscaling. It will also immediately show *why* pure spectral extrapolation isn't enough (ringing/Gibbs artifacts at edges, since real images aren't truly bandlimited).

## Phase 2 — Edge-directed interpolation

Estimate local gradient/covariance direction per missing pixel and interpolate *along* edges instead of across them. Fixes most "blurry diagonal edge" bicubic artifacts.
- NEDI (New Edge-Directed Interpolation)
- ICBI (Iterative Curvature-Based Interpolation) as a second variant to compare

## Phase 3 — Self-similarity / internal example-based SR

Strongest classical technique. Natural images have recurring patches across scales (a grass patch resembles a downscaled grass patch elsewhere in the same image). Build an internal multi-scale patch database from the LR image itself, and for each output patch, find the best-matching internal example to borrow high-frequency detail from.
- Based on Glasner et al., "Super-Resolution from a Single Image"
- No external training data needed
- Accelerate patch search with PatchMatch or a KD-tree/FLANN

## Phase 3.5 — Fractal-Based Upscaling (Iterated Function Systems)

Extends self-similarity by representing image regions not just as pixel patches, but as mathematical affine transformations (Iterated Function Systems). Under the assumption that natural images are fractals formed by self-transformations, this models mappings between large and small blocks to allow infinite analytical resolution scaling.
- Mathematically rigorous alternative to pure patch copying.
- Models scale, rotation, and contrast adjustments in the self-similarity search.

## Phase 4 — Sparse coding / dictionary-based SR

Best cost/quality tradeoff; bridges classical and learned methods. Train coupled low-res/high-res dictionaries once (offline, cheap) via K-SVD. Represent each LR patch as a sparse combination of low-res atoms, reconstruct using corresponding high-res atoms.
- Based on Yang et al., sparse representation SR
- Tiny, interpretable, CPU-cheap — not remotely VLM-scale compute

## Phase 4.5 — Hybrid vector/raster decomposition (SVG-style)

**Why it's a real option:** a raster image is resolution-bound by its pixel count. A vector representation (paths, gradients, primitives) is resolution-independent — re-rasterize at any scale with zero interpolation artifacts. This is exactly why it works well for logos, line art, text, UI screenshots, and flat-color illustration.

**Why it's not a universal method:** vectorization assumes the image decomposes into a small number of smooth regions with clean boundaries. Natural photos violate this — texture, noise, and stochastic gradients (skin, foliage, fabric, clouds) either explode the vector/path count until it's an inefficient re-encoding of pixels, or force smoothing aggressive enough to lose the image's essence.

**Where it fits here:** gated by a content-type classifier (flat-color-ratio or edge-density heuristic is enough), not applied uniformly:
1. Segment the image (superpixels or flat-region detector)
2. Classify regions as flat/smooth vs. textured
3. Vectorize flat regions — fit Bezier paths + gradient fills (Potrace-style curve fitting)
4. Leave textured regions to Phase 3/4 patch-based methods
5. Recombine at render time

**Stretch goal, not load-bearing:** diffusion curves / gradient mesh representations — generalize vector primitives to smooth-shaded regions with color diffusing from boundary curves. Closer mathematical match to photographic gradients than flat Bezier fills, but harder to fit and has less standard tooling.

**Caution:** do not make this the pipeline's anchor. Strong, cheap win for line art/text/logos/screenshots; a genuine dead end if applied by default to photographic content.

## Phase 5 — Wavelet-domain reconstruction

Decompose the LR image into wavelet subbands. Instead of just upsampling approximation coefficients, model/predict the missing high-frequency detail subbands (using the self-similarity or sparse-coding methods from Phases 3–4, applied per subband). Often cleaner than pixel-domain methods since it directly targets the missing high-frequency content.

## Phase 6 — Reconstruction-constrained refinement & TV Minimization

Take any candidate HR image from the phases above, simulate downsampling it, compare against the real LR input, and correct the HR estimate to reduce that residual. Enhance this refinement phase using **Partial Differential Equations (PDEs)** and **Total Variation (TV) Minimization** (e.g., the Rudin-Osher-Fatemi model). 
- Formulates upscaling as a rigorous optimization problem: *minimize the total variation (gradient integral) of the image subject to the downsampled version matching the input.*
- Perfectly preserves sharp edges while smoothing noise/artifacts mathematically.
- Iterative back-projection / POCS can be combined with Anisotropic Diffusion (Perona-Malik) for mathematically optimal edge preservation.

## Phase 7 — Combine and compare

Blend estimators per-patch (e.g., edge-directed for smooth edges, self-similarity/dictionary/vectorized for textured or flat regions respectively, chosen via a local variance/gradient/flatness heuristic). Re-run the Phase 0 harness across all combinations to find the best pure-math pipeline.

## Phase 7.5 — Continuous Mathematical Function Fitting (Implicit Representations)

Instead of treating the image as a discrete 2D grid of pixels, model it as a continuous, differentiable mathematical function: $f(x, y) \rightarrow (R, G, B)$. By fitting a network (like SIREN) or using Fourier Features, we mathematically parameterize the image's frequency domain.
- Bridges pure math (Fourier analysis) and continuous optimization.
- Once the continuous function is fit, you can query it at *any* arbitrary coordinate float value, achieving infinite continuous upscaling independent of pixel grids.

## Phase 8 — Deep Unfolding (Math + Lightweight AI Hybrid)

**Target:** Professional-grade, research-paper quality results.

This is the ultimate hybrid approach bridging rigorous linear algebra/inverse problems with a lightweight neural network (Algorithm Unrolling). We formulate upscaling as an inverse problem using Maximum A Posteriori (MAP) inference: $y = D H x + n$. 

Using Half-Quadratic Splitting (HQS) or ADMM, we unroll the optimization into iterative steps:
1. **Data Projection Step (Pure Math):** Analytically guarantees the HR image correctly downsamples back to the LR input (solved via FFT).
2. **Prior/Regularization Step (Lightweight AI):** Instead of a fixed mathematical prior (like TV), we insert a very small, fast denoising CNN (e.g., a 5-layer DnCNN) as a "learned proximal operator" to inject plausible high-frequency textures.

**Why this is the endgame:**
- **Research Paper Writable:** Deep Unfolding is a prestigious academic framework proving mathematical convergence while achieving AI-level perceptual quality.
- **Professional Results:** It produces incredibly sharp, hallucination-free output competitive with professional software.
- **Lightweight:** The AI component is tiny because it only acts as a denoiser within the math loop, rather than doing the entire heavy lifting of upscaling.

---

## Implementation note

Every phase/function should be independently swappable and testable against the Phase 0 harness: `numpy array in, numpy array out`. This lets any combination of methods be A/B tested without rewrites.
