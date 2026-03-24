import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis


def extract_spectral_features(image_rgb, k_dct=128, bins_dft=64):
    """
    Stream B: Frequency Spectral Analysis.
    Returns a 192-dimensional vector (128 DCT + 64 DFT).
    """
    # Convert to Luminance (Y) as specified in Section 6.2
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0].astype(np.float32)

    # 1. 2D Discrete Cosine Transform
    # Compute DCT and take the log-scaled absolute values
    dct_2d = dct(dct(y_channel.T, norm="ortho").T, norm="ortho")
    dct_log = np.log(np.abs(dct_2d) + 1e-6)

    # Extract top-k coefficients (zig-zag or flattened magnitude)
    # We take the largest magnitudes to capture grid artifacts
    dct_vector = np.sort(dct_log.flatten())[::-1][:k_dct]

    # 2. 2D Discrete Fourier Transform
    dft = np.fft.fft2(y_channel)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    # Radially Averaged Power Spectrum (Section 6.2)
    h, w = magnitude_spectrum.shape
    yc, xc = h // 2, w // 2
    y, x = np.ogrid[-yc : h - yc, -xc : w - xc]
    r = np.sqrt(x * x + y * y).astype(np.int32)

    # Bin the radial distances into bins_dft
    r_flatten = r.flatten()
    mag_flatten = magnitude_spectrum.flatten()
    radial_sum = np.bincount(r_flatten, weights=mag_flatten)
    radial_counts = np.bincount(r_flatten)

    # Ensure we return exactly 64 bins
    full_radial_vector = radial_sum / (radial_counts + 1e-6)
    dft_vector = full_radial_vector[1 : bins_dft + 1]
    if len(dft_vector) < bins_dft:
        dft_vector = np.pad(dft_vector, (0, bins_dft - len(dft_vector)))
    final_vector = np.concatenate([dct_vector, dft_vector])
    final_vector = np.log1p(final_vector)

    return final_vector.astype(np.float32)


def extract_depth_features(image_rgb, depth_map):
    """
    Stream C: Depth-RGB Discrepancy.
    Returns a 9-dimensional statistical vector.
    """
    # 1. Gradient Magnitude Extraction (Sobel)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_rgb = np.sqrt(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2
    )

    # Resize depth map back to image size if necessary
    depth_res = cv2.resize(depth_map, (grad_rgb.shape[1], grad_rgb.shape[0]))
    grad_depth = np.sqrt(
        cv2.Sobel(depth_res, cv2.CV_64F, 1, 0) ** 2
        + cv2.Sobel(depth_res, cv2.CV_64F, 0, 1) ** 2
    )

    # 2. Local Correlation / Discrepancy (Simplified per-pixel diff)
    # Quantify photometric-geometric inconsistencies (Section 4.2)
    diff = np.abs(grad_rgb - grad_depth)

    # 3. 9-Dimensional Feature Vector (Section 8)
    # Mean, Variance, Skewness, Kurtosis, and Percentiles (5, 25, 75, 95) + Median
    stats = [
        np.mean(diff),
        np.var(diff),
        skew(diff.flatten()),
        kurtosis(diff.flatten()),
        np.percentile(diff, 5),
        np.percentile(diff, 25),
        np.percentile(diff, 50),  # Median to fulfill the 9th dimension
        np.percentile(diff, 75),
        np.percentile(diff, 95),
    ]

    return np.array(stats, dtype=np.float32)
