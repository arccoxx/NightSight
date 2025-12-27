# NightSight Enhancement Methods Comparison

This directory contains comparison results for all available enhancement methods.

## Test Setup
- **Number of test images**: 3
- **Source**: LOL Dataset (eval15/low)
- **Device**: cuda
- **Date**: 2025-12-26 20:52:16

## Directory Structure

### 1_traditional/
Traditional (non-deep learning) image enhancement methods:
- **histogram_equalization**: Standard histogram equalization
- **clahe**: Contrast Limited Adaptive Histogram Equalization
- **gamma_mean**: Adaptive gamma correction using mean luminance
- **gamma_median**: Adaptive gamma correction using median luminance
- **retinex**: Multi-Scale Retinex with Color Restoration
- **bilateral**: Bilateral filtering for noise reduction

### 2_deep_learning/
Deep learning-based enhancement methods:
- **nightsight_trained**: Trained NightSightNet model (best checkpoint)
- **zerodce_untrained**: Zero-Reference Deep Curve Estimation (untrained baseline)

### 3_comparisons/
Side-by-side comparison grids showing all methods on the same image.

## Methods Overview

### Traditional Methods

**Histogram Equalization**
- Simple global contrast enhancement
- Fast but can over-enhance
- Good for uniform lighting

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Local adaptive contrast enhancement
- Prevents over-amplification of noise
- Best for images with varying lighting

**Adaptive Gamma Correction**
- Automatically adjusts gamma based on image statistics
- Mean method: Uses average brightness
- Median method: More robust to outliers

**Multi-Scale Retinex**
- Simulates human visual perception
- Separates illumination from reflectance
- Preserves color while enhancing details

**Bilateral Filter**
- Edge-preserving smoothing
- Reduces noise while maintaining sharpness
- Good for preprocessing or postprocessing

### Deep Learning Methods

**NightSightNet (Trained)**
- Hybrid architecture combining multiple techniques
- Trained on LOL dataset for 200 epochs
- Best PSNR: 20.10 dB, SSIM: 0.95
- Real-time capable: ~65 FPS at VGA resolution

**Zero-DCE (Untrained)**
- Zero-reference learning approach
- Lightweight model for fast inference
- Shown here as untrained baseline for comparison

## Performance Comparison

See individual comparison grids in `3_comparisons/` for visual quality assessment.

## Usage

To reproduce these results:
```bash
python scripts/test_all_methods.py
```

To test real-time enhancement:
```bash
python scripts/realtime_demo.py --camera 0
```

## Notes

- All images are from the LOL dataset validation set
- Traditional methods are parameter-tuned for low-light scenarios
- Deep learning models use GPU acceleration when available
- Processing times may vary based on image size and hardware
