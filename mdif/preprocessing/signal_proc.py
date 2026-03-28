"""
Implements the core signal processing functions for two of the three streams of the proposed method:
- Stream B: Frequency Spectral Analysis (DCT and DFT features)
- Stream C: Depth-RGB Discrepancy (gradient-based features and statistical descriptors)
"""

import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis


def extract_spectral_features(image_rgb, k_dct=128, bins_dft=64) -> np.ndarray:
    """
    Stream B: Frequency Spectral Analysis.

    Extracts a combined feature vector from the input RGB image using:
    1. Discrete Cosine Transform (DCT): Top k coefficients from the log-scaled DCT of the luminance channel.
    2. Discrete Fourier Transform (DFT): Radially averaged power spectrum from the DFT magnitude.

    :param image_rgb: Input image in RGB format (H x W x 3)
    :param k_dct: Number of top DCT coefficients to keep
    :param bins_dft: Number of bins for the DFT radial average

    :return: 192-dimensional feature vector (128 DCT + 64 DFT).
    :rtype: np.ndarray
    """

    # Convert to Luminance (Y) for frequency analysis
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0].astype(np.float32)

    # 2D Discrete Cosine Transform
    # Compute DCT and take the log-scaled absolute values
    dct_2d = dct(dct(y_channel.T, norm="ortho").T, norm="ortho")
    dct_log = np.log(np.abs(dct_2d) + 1e-6)

    dct_vector = np.sort(dct_log.flatten())[::-1][:k_dct]

    # 2D Discrete Fourier Transform
    dft = np.fft.fft2(y_channel)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    # Radially Averaged Power Spectrum
    h, w = magnitude_spectrum.shape
    yc, xc = h // 2, w // 2
    y, x = np.ogrid[-yc : h - yc, -xc : w - xc]
    r = np.sqrt(x * x + y * y).astype(np.int32)

    r_flatten = r.flatten()
    mag_flatten = magnitude_spectrum.flatten()
    radial_sum = np.bincount(r_flatten, weights=mag_flatten)
    radial_counts = np.bincount(r_flatten)

    full_radial_vector = radial_sum / (radial_counts + 1e-6)
    dft_vector = full_radial_vector[1 : bins_dft + 1]
    if len(dft_vector) < bins_dft:
        dft_vector = np.pad(dft_vector, (0, bins_dft - len(dft_vector)))
    final_vector = np.concatenate([dct_vector, dft_vector])
    final_vector = np.log1p(final_vector)

    return final_vector.astype(np.float32)


def extract_depth_features(image_rgb, depth_map) -> np.ndarray:
    """
    Stream C: Depth-RGB Discrepancy.

    Computes a 9-dimensional feature vector that captures the local correlation and discrepancy between the RGB image and its corresponding depth map. The features include:
    1. Gradient Magnitude Extraction: Computes the gradient magnitude for both the RGB image (using the luminance channel) and the depth map using Sobel operators.
    2. Local Correlation / Discrepancy: Calculates the absolute difference between the RGB and depth gradients to capture local inconsistencies.
    3. Statistical Descriptors: Computes mean, variance, skewness, kurtosis, and percentiles (5th, 25th, 50th, 75th, 95th) of the discrepancy map to summarize the distribution of discrepancies.

    :param image_rgb: Input RGB image (H x W x 3)
    :param depth_map: Corresponding depth map (H x W)

    :return: 9-dimensional statistical vector.
    :rtype: np.ndarray
    """

    # Gradient Magnitude Extraction (Sobel)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_rgb = np.sqrt(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2
    )

    depth_res = cv2.resize(depth_map, (grad_rgb.shape[1], grad_rgb.shape[0]))
    grad_depth = np.sqrt(
        cv2.Sobel(depth_res, cv2.CV_64F, 1, 0) ** 2
        + cv2.Sobel(depth_res, cv2.CV_64F, 0, 1) ** 2
    )

    # Local Correlation / Discrepancy (Simplified per-pixel diff)
    diff = np.abs(grad_rgb - grad_depth)

    # 9-Dimensional Feature Vector
    # Mean, Variance, Skewness, Kurtosis, and Percentiles (5, 25, 75, 95) + Median
    stats = [
        np.mean(diff),
        np.var(diff),
        skew(diff.flatten()),
        kurtosis(diff.flatten()),
        np.percentile(diff, 5),
        np.percentile(diff, 25),
        np.percentile(diff, 50),
        np.percentile(diff, 75),
        np.percentile(diff, 95),
    ]

    return np.array(stats, dtype=np.float32)
