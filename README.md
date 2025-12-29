# NightSight

**Advanced Night Vision Enhancement from Standard Cameras Using Deep Learning**

NightSight is a comprehensive Python package for enhancing low-light and nighttime images using state-of-the-art deep learning techniques combined with traditional image processing methods. It enables "seeing in the dark" without specialized hardware, extracting maximum information from severely underexposed images.

## Sample Results

![NightSight v1 Enhancement Results](outputs/method_comparison/3_comparisons/146_all_methods.png)

*Comparison of v1 methods showing Input, Traditional (SSR, MSR, MSRCR, CLAHE), Deep Learning (Zero-DCE, NightSightNet), and Ground Truth.*

## Features

- **Multiple Deep Learning Models**
  - Zero-DCE: Zero-reference curve estimation for self-supervised enhancement
  - Retinexformer: Transformer-based Retinex decomposition
  - SwinIR: Swin Transformer for image restoration
  - Diffusion Models: Physics-guided diffusion for high-quality enhancement
  - NightSightNet: Our hybrid model combining multiple techniques

- **Traditional Processing**
  - Retinex algorithms (SSR, MSR, MSRCR)
  - CLAHE and adaptive histogram equalization
  - Bilateral and guided filtering
  - Wavelet denoising
  - Edge detection and enhancement

- **Advanced Capabilities**
  - Multi-frame temporal fusion for video
  - Physics-based noise modeling
  - Real-time processing support
  - RAW image processing

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/arccoxx/NightSight.git
cd nightsight
pip install -e .
```

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python >= 3.8
- PyTorch >= 2.0
- OpenCV >= 4.5
- NumPy >= 1.21

## Quick Start

### Enhance a Single Image

```python
from nightsight.pipelines import SingleImagePipeline

# Create pipeline
pipeline = SingleImagePipeline()

# Enhance image
enhanced = pipeline.enhance("dark_image.jpg", output_path="enhanced.jpg")
```

### Using Traditional Methods

```python
from nightsight.traditional import RetinexEnhancer, CLAHEEnhancer

# Retinex enhancement
retinex = RetinexEnhancer(method="msrcr")
enhanced = retinex.enhance(image)

# CLAHE enhancement
clahe = CLAHEEnhancer(clip_limit=3.0)
enhanced = clahe.enhance(image)
```

### Using Deep Learning Models

```python
from nightsight.models import ZeroDCE, Retinexformer, NightSightNet
import torch

# Load model
model = NightSightNet()
model.load_pretrained("checkpoints/best_model.pth")
model.eval()

# Enhance
with torch.no_grad():
    enhanced = model(low_light_tensor)
```

### Video Enhancement

```python
from nightsight.pipelines import VideoPipeline

# Create pipeline with temporal fusion
pipeline = VideoPipeline(use_temporal=True)

# Enhance video
pipeline.enhance_video("dark_video.mp4", "enhanced_video.mp4")
```

### Real-time Webcam Demo

```python
from nightsight.pipelines.video import RealtimePipeline

pipeline = RealtimePipeline()
pipeline.run_webcam()
```

## Training

### Prepare Dataset

NightSight supports multiple dataset formats:

1. **LOL Dataset** (recommended):
   ```
   data/LOL/
   ├── our485/
   │   ├── low/
   │   └── high/
   └── eval15/
       ├── low/
       └── high/
   ```

2. **Paired Dataset**:
   ```
   data/
   ├── train/
   │   ├── low/
   │   └── high/
   └── val/
       ├── low/
       └── high/
   ```

3. **Synthetic Dataset** (from clean images):
   ```
   data/clean/
   └── *.jpg
   ```

### Train a Model

```bash
# Train NightSight model
python scripts/train.py --model nightsight --data-dir data/LOL --epochs 200

# Train Zero-DCE
python scripts/train.py --model zerodce --data-dir data/LOL

# Train with custom config
python scripts/train.py --config configs/train_nightsight.yaml
```

### Monitor Training

```bash
tensorboard --logdir outputs/
```

## Inference

### Command Line

```bash
# Single image
python scripts/inference.py -i dark.jpg -o enhanced.jpg

# Multiple images
python scripts/inference.py -i input_folder/ -o output_folder/

# Video
python scripts/inference.py -i dark_video.mp4 -o enhanced_video.mp4

# With specific model
python scripts/inference.py -i image.jpg -o out.jpg -c checkpoints/best.pth
```

### Demo Script

```bash
# Compare all methods
python scripts/demo.py --compare dark_image.jpg

# Live webcam demo
python scripts/demo.py --webcam
```

## Architecture

```
nightsight/
├── core/              # Base classes and registry
├── config/            # Configuration management
├── data/              # Datasets and transforms
├── traditional/       # Classical image processing
│   ├── retinex.py     # Retinex algorithms
│   ├── histogram.py   # Histogram enhancement
│   ├── filters.py     # Bilateral, guided filters
│   ├── frequency.py   # FFT, wavelet processing
│   ├── edge.py        # Edge detection
│   └── motion.py      # Optical flow, motion
├── models/            # Deep learning models
│   ├── zerodce.py     # Zero-DCE
│   ├── retinexformer.py
│   ├── swinir.py
│   ├── unet.py
│   ├── diffusion/     # Diffusion models
│   └── hybrid.py      # NightSightNet
├── temporal/          # Multi-frame processing
│   ├── alignment.py   # Frame alignment
│   └── fusion.py      # Temporal fusion
├── physics/           # Physics-based models
│   ├── noise.py       # Noise modeling
│   └── illumination.py
├── losses/            # Loss functions
├── metrics/           # Evaluation metrics
├── utils/             # Utilities
└── pipelines/         # High-level pipelines
```

## Methods Overview

### Retinex Theory

NightSight implements Retinex-based decomposition:

```
I = R × L
```

Where `I` is the observed image, `R` is reflectance, and `L` is illumination.

### Zero-DCE

Zero-Reference Deep Curve Estimation learns image-specific tone curves:

```
LE(x) = x + α × x × (1 - x)
```

Applied iteratively with learned α parameters.

### Diffusion Models

Physics-guided diffusion with Retinex constraints for accurate enhancement without hallucinations.

### Temporal Fusion

Multi-frame processing with:
- Optical flow alignment
- Deformable convolutions
- Attention-based fusion
- Recurrent processing for video

## Benchmarks

Performance on LOL dataset:

| Model | PSNR | SSIM | Params | FPS (GPU) |
|-------|------|------|--------|-----------|
| Zero-DCE | 21.8 | 0.81 | 79K | 120+ |
| Retinexformer | 23.5 | 0.85 | 1.6M | 45 |
| NightSightNet | 24.2 | 0.87 | 2.1M | 35 |

---

## 🧪 NightSight v2 (Experimental)

**⚠️ EXPERIMENTAL:** NightSight v2 is an experimental version inspired by military night vision systems. While it includes interesting features like depth-aware outlines and object tracking, **v1 (above) provides better overall image quality for most use cases**.

v2 is recommended only for:
- Experimental/research purposes
- Real-time object tracking applications
- Military night vision aesthetic preferences

### v2 Features

- 🎯 Depth-based object differentiation
- ✨ Glowing edge outlines (military night vision style)
- 🔍 Real-time object detection & tracking (YOLOv8n)
- 📊 Scene-adaptive processing
- 🚀 Zero-DCE++ low-light enhancement

### v2 Sample Results

![v1 vs v2 Comparison](outputs/readme_samples/146_comparison.png)

*Comparison: Original | v1 (Recommended) | v2 (Experimental)*

**Note:** v2 is optimized for real-time tracking and stylistic effects rather than pure image quality.

### v2 Quick Start

```python
from nightsight.v2 import NightSightV2Pipeline

# Create v2 pipeline
pipeline = NightSightV2Pipeline(device='cuda')

# Enhance image
enhanced = pipeline.enhance_image('dark.jpg', 'enhanced_v2.jpg')

# Real-time webcam with tracking
pipeline.process_webcam(camera_id=0)
```

### v2 Command Line

```bash
# Process an image
python scripts/inference_v2.py -i dark.jpg -o enhanced.jpg

# Real-time webcam demo
python scripts/realtime_v2_demo.py --camera 0

# Compare v1 vs v2
python scripts/demo_v1_v2_comparison.py -i dark.jpg
```

**[See README_V2.md for full v2 documentation](README_V2.md)**

---

## Citation

If you use NightSight in your research, please cite:

```bibtex
@software{nightsight2024,
  title={NightSight: Advanced Night Vision Enhancement},
  author={NightSight Team},
  year={2024},
  url={https://github.com/arccoxx/NightSight}
}
```

## Related Papers

This implementation draws inspiration from:

- Zero-DCE: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" (CVPR 2020)
- Retinexformer: "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" (ICCV 2023)
- SwinIR: "SwinIR: Image Restoration Using Swin Transformer" (ICCV 2021)
- NTIRE 2024 Low Light Enhancement Challenge

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## Acknowledgments

- LOL Dataset authors
- PyTorch team
- NTIRE challenge organizers
