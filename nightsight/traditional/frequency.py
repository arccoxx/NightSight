"""
Frequency domain filtering for low-light enhancement.

Includes FFT filtering, wavelet decomposition, and homomorphic filtering.
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional


def fft_filter(
    image: np.ndarray,
    filter_type: str = "highpass",
    cutoff: float = 30,
    order: int = 2
) -> np.ndarray:
    """
    Apply FFT-based frequency filtering.

    Args:
        image: Input image (grayscale, float32 [0, 1])
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff: Cutoff frequency (or tuple for bandpass)
        order: Filter order (for Butterworth)

    Returns:
        Filtered image
    """
    if len(image.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = fft_filter(image[:, :, c], filter_type, cutoff, order)
        return result

    # Compute FFT
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Create filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create distance matrix
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    d = np.sqrt(u ** 2 + v ** 2)

    # Create filter based on type
    if filter_type == "lowpass":
        # Butterworth lowpass
        h = 1 / (1 + (d / cutoff) ** (2 * order))

    elif filter_type == "highpass":
        # Butterworth highpass
        h = 1 / (1 + (cutoff / (d + 1e-10)) ** (2 * order))

    elif filter_type == "bandpass":
        # Band-pass filter
        low_cut, high_cut = cutoff if isinstance(cutoff, tuple) else (cutoff * 0.5, cutoff * 1.5)
        h_low = 1 / (1 + (d / low_cut) ** (2 * order))
        h_high = 1 / (1 + (high_cut / (d + 1e-10)) ** (2 * order))
        h = h_low * h_high

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply filter
    filtered_dft = dft_shift * h

    # Inverse FFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    filtered = np.fft.ifft2(dft_ishift)
    filtered = np.abs(filtered)

    return filtered.astype(np.float32)


def homomorphic_filter(
    image: np.ndarray,
    gamma_low: float = 0.5,
    gamma_high: float = 2.0,
    cutoff: float = 30,
    order: int = 2
) -> np.ndarray:
    """
    Apply homomorphic filtering for illumination correction.

    Homomorphic filtering works in the frequency domain to separate
    illumination (low frequency) from reflectance (high frequency),
    allowing independent enhancement of each component.

    Args:
        image: Input image (float32, [0, 1])
        gamma_low: Gain for low frequencies (illumination)
        gamma_high: Gain for high frequencies (reflectance)
        cutoff: Cutoff frequency for the filter
        order: Filter order

    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Process each channel
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = homomorphic_filter(
                image[:, :, c], gamma_low, gamma_high, cutoff, order
            )
        return result

    # Take log to convert multiplicative to additive
    log_image = np.log(image + 1e-6)

    # Compute FFT
    dft = np.fft.fft2(log_image)
    dft_shift = np.fft.fftshift(dft)

    # Create homomorphic filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    d = np.sqrt(u ** 2 + v ** 2)

    # High-pass filter with gamma adjustment
    h = (gamma_high - gamma_low) * (1 - np.exp(-order * (d ** 2) / (cutoff ** 2))) + gamma_low

    # Apply filter
    filtered_dft = dft_shift * h

    # Inverse FFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    filtered_log = np.fft.ifft2(dft_ishift)
    filtered_log = np.real(filtered_log)

    # Exponential to reverse the log
    filtered = np.exp(filtered_log) - 1e-6

    # Normalize
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)

    return filtered.astype(np.float32)


def wavelet_decompose(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1
) -> Tuple[np.ndarray, list]:
    """
    Decompose image using 2D wavelet transform.

    Args:
        image: Input image (grayscale or color)
        wavelet: Wavelet type ('haar', 'db1', etc.)
        level: Decomposition level

    Returns:
        Tuple of (approximation coefficients, list of detail coefficients)
    """
    try:
        import pywt
    except ImportError:
        # Fallback to simple Haar implementation
        return _haar_decompose(image, level)

    if len(image.shape) == 3:
        # Process each channel
        results = []
        for c in range(image.shape[2]):
            coeffs = pywt.wavedec2(image[:, :, c], wavelet, level=level)
            results.append(coeffs)
        # Combine results
        return results
    else:
        return pywt.wavedec2(image, wavelet, level=level)


def _haar_decompose(
    image: np.ndarray,
    level: int = 1
) -> Tuple[np.ndarray, list]:
    """Simple Haar wavelet decomposition."""
    result_details = []

    for _ in range(level):
        h, w = image.shape[:2]
        h = h - h % 2
        w = w - w % 2
        image = image[:h, :w]

        # Row-wise transform
        avg = (image[:, 0::2] + image[:, 1::2]) / 2
        diff_h = (image[:, 0::2] - image[:, 1::2]) / 2

        # Column-wise transform
        ll = (avg[0::2, :] + avg[1::2, :]) / 2
        lh = (avg[0::2, :] - avg[1::2, :]) / 2
        hl = (diff_h[0::2, :] + diff_h[1::2, :]) / 2
        hh = (diff_h[0::2, :] - diff_h[1::2, :]) / 2

        result_details.append((lh, hl, hh))
        image = ll

    return image, result_details


def wavelet_reconstruct(
    coeffs: Tuple,
    wavelet: str = "haar"
) -> np.ndarray:
    """
    Reconstruct image from wavelet coefficients.

    Args:
        coeffs: Wavelet coefficients (from wavelet_decompose)
        wavelet: Wavelet type

    Returns:
        Reconstructed image
    """
    try:
        import pywt
        return pywt.waverec2(coeffs, wavelet)
    except ImportError:
        return _haar_reconstruct(coeffs)


def _haar_reconstruct(coeffs: Tuple) -> np.ndarray:
    """Simple Haar wavelet reconstruction."""
    ll, details = coeffs

    for lh, hl, hh in reversed(details):
        h, w = ll.shape[:2]

        # Column-wise inverse
        avg_row = np.zeros((h * 2, w))
        avg_row[0::2, :] = ll + lh
        avg_row[1::2, :] = ll - lh

        diff_row = np.zeros((h * 2, w))
        diff_row[0::2, :] = hl + hh
        diff_row[1::2, :] = hl - hh

        # Row-wise inverse
        ll = np.zeros((h * 2, w * 2))
        ll[:, 0::2] = avg_row + diff_row
        ll[:, 1::2] = avg_row - diff_row

    return ll


def wavelet_denoise(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    threshold_type: str = "soft",
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Apply wavelet-based denoising.

    Args:
        image: Input image (float32, [0, 1])
        wavelet: Wavelet type
        level: Decomposition level
        threshold_type: 'soft' or 'hard' thresholding
        threshold: Threshold value (auto-computed if None)

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = wavelet_denoise(
                image[:, :, c], wavelet, level, threshold_type, threshold
            )
        return result

    # Decompose
    try:
        import pywt

        coeffs = pywt.wavedec2(image, wavelet, level=level)

        # Estimate noise and compute threshold
        if threshold is None:
            # Use MAD estimator on finest scale
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(image.size))

        # Apply thresholding to detail coefficients
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_detail = []
            for d in detail:
                if threshold_type == "soft":
                    thresholded = pywt.threshold(d, threshold, mode='soft')
                else:
                    thresholded = pywt.threshold(d, threshold, mode='hard')
                new_detail.append(thresholded)
            new_coeffs.append(tuple(new_detail))

        # Reconstruct
        denoised = pywt.waverec2(new_coeffs, wavelet)

    except ImportError:
        # Fallback without pywt
        ll, details = _haar_decompose(image, level)

        if threshold is None:
            sigma = np.median(np.abs(details[-1][0])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(image.size))

        new_details = []
        for lh, hl, hh in details:
            if threshold_type == "soft":
                lh = np.sign(lh) * np.maximum(np.abs(lh) - threshold, 0)
                hl = np.sign(hl) * np.maximum(np.abs(hl) - threshold, 0)
                hh = np.sign(hh) * np.maximum(np.abs(hh) - threshold, 0)
            else:
                lh = lh * (np.abs(lh) > threshold)
                hl = hl * (np.abs(hl) > threshold)
                hh = hh * (np.abs(hh) > threshold)
            new_details.append((lh, hl, hh))

        denoised = _haar_reconstruct((ll, new_details))

    # Ensure output matches input size and range
    denoised = denoised[:image.shape[0], :image.shape[1]]
    denoised = np.clip(denoised, 0, 1)

    return denoised.astype(np.float32)


def frequency_decomposition(
    image: np.ndarray,
    num_bands: int = 3
) -> list:
    """
    Decompose image into frequency bands using Laplacian pyramid.

    Args:
        image: Input image
        num_bands: Number of frequency bands

    Returns:
        List of frequency bands (high to low frequency)
    """
    bands = []
    current = image.astype(np.float32)

    for i in range(num_bands - 1):
        # Downsample
        down = cv2.pyrDown(current)
        # Upsample back
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        # Difference is high frequency
        high_freq = current - up
        bands.append(high_freq)
        current = down

    # Last band is low frequency
    bands.append(current)

    return bands


def frequency_reconstruction(bands: list) -> np.ndarray:
    """
    Reconstruct image from frequency bands.

    Args:
        bands: List of frequency bands (from frequency_decomposition)

    Returns:
        Reconstructed image
    """
    # Start with lowest frequency
    current = bands[-1]

    for high_freq in reversed(bands[:-1]):
        # Upsample
        current = cv2.pyrUp(current, dstsize=(high_freq.shape[1], high_freq.shape[0]))
        # Add high frequency
        current = current + high_freq

    return current
