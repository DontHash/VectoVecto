"""
tv_refinement.py — Phase 6: Total Variation (TV) Minimization & ROF Refinement.

Formulates image refinement as a mathematically rigorous optimization problem:
    min_x  0.5 * || D H x - y ||_2^2  +  lambda_tv * TV(x)

Where:
  - D H x is the simulated downsampled projection of candidate HR image x.
  - y is the original LR input ground truth.
  - TV(x) is the isotropic Total Variation (L1 norm of gradient magnitude),
    which penalizes noise/artifacts while strictly preserving sharp discontinuous edges.

Solved via Half-Quadratic Splitting / Iterative Back-Projection with Chambolle's Dual TV Proximal operator.
API: numpy in -> numpy out (compliant with Phase 0 evaluation harness).
"""
import numpy as np
import cv2

def compute_divergence(p):
    """
    Computes divergence of a 2D vector field p = (px, py).
    Adjoint operator to the gradient: div(p) = d(px)/dx + d(py)/dy
    p shape: (2, H, W)
    """
    px, py = p[0], p[1]
    H, W = px.shape
    div = np.zeros((H, W), dtype=px.dtype)

    div[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    div[:, 0] += px[:, 0]
    div[:, -1] -= px[:, -2]

    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[0, :] += py[0, :]
    div[-1, :] -= py[-2, :]

    return div

def compute_gradient(img):
    """
    Computes forward difference gradient: grad(u) = (du/dx, du/dy).
    img shape: (H, W)
    returns: (2, H, W)
    """
    H, W = img.shape
    grad = np.zeros((2, H, W), dtype=img.dtype)
    grad[0, :, :-1] = img[:, 1:] - img[:, :-1]
    grad[1, :-1, :] = img[1:, :] - img[:-1, :]
    return grad

def chambolle_tv_denoise(img, weight=0.1, max_iter=25):
    """
    Solves the Rudin-Osher-Fatemi (ROF) TV denoising model:
        min_u  0.5 * ||u - f||_2^2 + weight * TV(u)
    using Chambolle's fast dual projection algorithm.
    img: 2D numpy array [0, 1]
    weight: regularization parameter (lambda)
    """
    if weight <= 0.0 or max_iter <= 0:
        return img

    H, W = img.shape
    p = np.zeros((2, H, W), dtype=np.float32)
    tau = 0.24

    for _ in range(max_iter):
        div_p = compute_divergence(p)
        grad_div = compute_gradient(div_p - img / weight)
        norm_grad = np.maximum(1.0, np.sqrt(grad_div[0]**2 + grad_div[1]**2) * tau)
        p = (p + tau * grad_div) / norm_grad

    return img - weight * compute_divergence(p)

def tv_super_resolution_refine(lr_img, hr_estimate, scale=4, lambda_tv=0.03,
                               ibp_weight=0.2, num_iters=6, blur_sigma=1.2):
    """
    Phase 6 Reconstruction-Constrained TV Minimization:
    Iterative Back-Projection (IBP) combined with Chambolle's TV proximal operator.

    lr_img: Original low-res input (H_lr, W_lr, C) or (H_lr, W_lr) in uint8 or [0, 1]
    hr_estimate: Candidate high-res image to refine (H_hr, W_hr, C) in uint8 or [0, 1]
    scale: upscaling factor
    lambda_tv: strength of TV smoothing on flat regions
    ibp_weight: step size for data fidelity projection
    num_iters: number of outer IBP-TV optimization iterations
    """
    is_uint8 = (hr_estimate.dtype == np.uint8)
    lr = lr_img.astype(np.float32) / 255.0 if lr_img.dtype == np.uint8 else lr_img.copy().astype(np.float32)
    hr = hr_estimate.astype(np.float32) / 255.0 if is_uint8 else hr_estimate.copy().astype(np.float32)

    has_channels = (len(hr.shape) == 3)
    num_channels = hr.shape[2] if has_channels else 1
    if not has_channels:
        hr = hr[:, :, np.newaxis]
        lr = lr[:, :, np.newaxis]

    kernel_size = int(np.ceil(blur_sigma * 3) * 2 + 1)
    k1d = cv2.getGaussianKernel(kernel_size, blur_sigma)
    k2d = np.outer(k1d, k1d).astype(np.float32)

    out = hr.copy()

    for it in range(num_iters):
        cur_lambda = lambda_tv * (0.8 ** it)

        for c in range(num_channels):
            cur_ch = out[:, :, c]
            lr_ch = lr[:, :, c]

            blurred = cv2.filter2D(cur_ch, -1, k2d, borderType=cv2.BORDER_REFLECT)
            downsampled = blurred[::scale, ::scale]

            min_h = min(downsampled.shape[0], lr_ch.shape[0])
            min_w = min(downsampled.shape[1], lr_ch.shape[1])

            residual = np.zeros_like(downsampled)
            residual[:min_h, :min_w] = downsampled[:min_h, :min_w] - lr_ch[:min_h, :min_w]

            dt_residual = np.zeros_like(cur_ch)
            dt_residual[::scale, ::scale] = residual
            grad_fidelity = cv2.filter2D(dt_residual, -1, k2d, borderType=cv2.BORDER_REFLECT)

            x_step = cur_ch - ibp_weight * grad_fidelity
            x_step = np.clip(x_step, 0.0, 1.0)

            x_tv = chambolle_tv_denoise(x_step, weight=cur_lambda, max_iter=15)
            out[:, :, c] = np.clip(x_tv, 0.0, 1.0)

    if not has_channels:
        out = out[:, :, 0]

    if is_uint8:
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    print("VectorScaling Phase 6 — Total Variation (TV) Minimization initialized.")
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:44, 20:44] = 1.0
    noisy = img + np.random.normal(0, 0.1, img.shape).astype(np.float32)
    denoised = chambolle_tv_denoise(noisy, weight=0.1)
    print(f"Self-test: noisy MSE={np.mean((noisy-img)**2):.4f} -> TV cleaned MSE={np.mean((denoised-img)**2):.4f}")
