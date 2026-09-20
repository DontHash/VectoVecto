import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.signal

def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    # Determine multichannel based on shape (RGB vs Grayscale)
    multichannel = True if len(img1.shape) == 3 else False
    
    # Starting with scikit-image 0.19, multichannel is deprecated in favor of channel_axis
    try:
        return ssim(img1, img2, data_range=255, channel_axis=-1 if multichannel else None)
    except TypeError:
        return ssim(img1, img2, data_range=255, multichannel=multichannel)

def degrade_image(hr_img, scale=2, blur_sigma=1.2):
    """
    Simulates the mathematical inverse problem: y = D H x
    H: Gaussian Blur (anti-aliasing)
    D: Subsampling (decimation)
    """
    # Create Gaussian kernel
    kernel_size = int(np.ceil(blur_sigma * 3) * 2 + 1)
    kernel_1d = cv2.getGaussianKernel(kernel_size, blur_sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Apply blur (H)
    if len(hr_img.shape) == 3:
        blurred = np.zeros_like(hr_img, dtype=np.float32)
        for c in range(3):
            blurred[:, :, c] = scipy.signal.convolve2d(hr_img[:, :, c], kernel_2d, mode='same', boundary='wrap')
    else:
        blurred = scipy.signal.convolve2d(hr_img, kernel_2d, mode='same', boundary='wrap')
        
    # Subsample (D)
    lr_img = blurred[::scale, ::scale]
    return np.clip(lr_img, 0, 255).astype(np.uint8)

def baseline_bicubic_upscale(lr_img, scale=2):
    """Simple cv2 bicubic baseline"""
    h, w = lr_img.shape[:2]
    return cv2.resize(lr_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def baseline_lanczos_upscale(lr_img, scale=2):
    """Proper Lanczos-4 resampling baseline"""
    h, w = lr_img.shape[:2]
    return cv2.resize(lr_img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

def naive_fft_upscale(lr_img, scale=2):
    """
    Phase 1: Naive FFT zero-padding upscaling.
    DFT the LR image, zero-pad in frequency, inverse transform.
    Ideal bandlimited interpolation.
    """
    h, w = lr_img.shape[:2]
    out_h, out_w = h * scale, w * scale
    
    # Handle channels if RGB
    is_rgb = len(lr_img.shape) == 3
    if is_rgb:
        channels = cv2.split(lr_img)
    else:
        channels = [lr_img]
        
    out_channels = []
    for ch in channels:
        # Compute FFT and shift zero frequency to center
        f = np.fft.fft2(ch)
        fshift = np.fft.fftshift(f)
        
        # Create zero-padded frequency domain image
        padded_fshift = np.zeros((out_h, out_w), dtype=np.complex128)
        
        # Place original frequencies in the center
        pad_y = (out_h - h) // 2
        pad_x = (out_w - w) // 2
        padded_fshift[pad_y:pad_y+h, pad_x:pad_x+w] = fshift
        
        # Inverse shift and IFFT
        f_ishift = np.fft.ifftshift(padded_fshift)
        img_back = np.fft.ifft2(f_ishift)
        
        # Magnitude and scale amplitude by scale^2
        img_back = np.real(img_back) * (scale ** 2)
        out_channels.append(np.clip(img_back, 0, 255).astype(np.uint8))
        
    if is_rgb:
        return cv2.merge(out_channels)
    else:
        return out_channels[0]

if __name__ == "__main__":
    print("VectorScaling Phase 0 & 1 Eval Harness initialized.")
    print("Generating synthetic test image (Zone Plate)...")
    
    # Create a synthetic Zone Plate image (rich in all frequencies)
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    hr_source = (np.sin(50 * np.pi * R**2) * 127.5 + 127.5).astype(np.uint8)
    
    scale = 2
    print(f"Simulating degradation (scale={scale}x)...")
    lr_simulated = degrade_image(hr_source, scale=scale)
    
    print("\nEvaluating Baselines:")
    
    # 1. Bicubic
    upscaled_bicubic = baseline_bicubic_upscale(lr_simulated, scale=scale)
    psnr_bicubic = calculate_psnr(hr_source, upscaled_bicubic)
    ssim_bicubic = calculate_ssim(hr_source, upscaled_bicubic)
    print(f"Bicubic Baseline -> PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}")
    
    # 2. Lanczos
    upscaled_lanczos = baseline_lanczos_upscale(lr_simulated, scale=scale)
    psnr_lanczos = calculate_psnr(hr_source, upscaled_lanczos)
    ssim_lanczos = calculate_ssim(hr_source, upscaled_lanczos)
    print(f"Lanczos Baseline -> PSNR: {psnr_lanczos:.2f} dB, SSIM: {ssim_lanczos:.4f}")
    
    # 3. Naive FFT
    upscaled_fft = naive_fft_upscale(lr_simulated, scale=scale)
    psnr_fft = calculate_psnr(hr_source, upscaled_fft)
    ssim_fft = calculate_ssim(hr_source, upscaled_fft)
    print(f"Naive FFT Upscale -> PSNR: {psnr_fft:.2f} dB, SSIM: {ssim_fft:.4f}")

    # 4. Phase 6: TV Minimization & ROF Refinement
    try:
        from tv_refinement import tv_super_resolution_refine
        upscaled_tv = tv_super_resolution_refine(lr_simulated, upscaled_bicubic, scale=scale, lambda_tv=0.02, num_iters=6)
        psnr_tv = calculate_psnr(hr_source, upscaled_tv)
        ssim_tv = calculate_ssim(hr_source, upscaled_tv)
        print(f"Phase 6 TV Minimization (ROF) -> PSNR: {psnr_tv:.2f} dB, SSIM: {ssim_tv:.4f}")
    except Exception as e:
        print(f"Phase 6 TV Minimization error: {e}")

    # 5. Phase 4.5: Hybrid Vector / Raster Decomposition
    try:
        from vector_raster_hybrid import hybrid_vector_raster_upscale
        upscaled_hybrid = hybrid_vector_raster_upscale(lr_simulated, scale=scale, raster_engine="bicubic")
        psnr_hybrid = calculate_psnr(hr_source, upscaled_hybrid)
        ssim_hybrid = calculate_ssim(hr_source, upscaled_hybrid)
        print(f"Phase 4.5 Hybrid Vector/Raster -> PSNR: {psnr_hybrid:.2f} dB, SSIM: {ssim_hybrid:.4f}")
    except Exception as e:
        print(f"Phase 4.5 Hybrid Vector/Raster error: {e}")

    # 6. Phase 7: Unified Autonomous Smart Router
    try:
        from smart_upscaler import smart_upscale
        upscaled_smart = smart_upscale(lr_simulated, scale=scale, mode="auto", fast=True)
        psnr_smart = calculate_psnr(hr_source, upscaled_smart)
        ssim_smart = calculate_ssim(hr_source, upscaled_smart)
        print(f"Phase 7 Smart Router Pipeline -> PSNR: {psnr_smart:.2f} dB, SSIM: {ssim_smart:.4f}")
    except Exception as e:
        print(f"Phase 7 Smart Router error: {e}")


