# NightSight Pre-trained Models

This directory contains the best trained model checkpoint.

## Available Models

### NightSightNet (Best Model)

**Files:**
- `nightsight_best.pth` - Model weights (2.1 MB)
- `nightsight_best.json` - Training metadata

**Performance:**
- **PSNR**: 20.10 dB
- **SSIM**: 0.9524
- **Training Epoch**: 156
- **Dataset**: LOL (Low-Light)

**Usage:**

```python
from nightsight.models.hybrid import NightSightNet
from nightsight.utils.checkpoint import load_checkpoint

# Load model
model = NightSightNet()
model, info = load_checkpoint('models/nightsight_best.pth', model, device='cuda')
model.eval()

# Use for inference
import torch
from PIL import Image
import numpy as np

# Load image
img = Image.open('dark_image.jpg')
img_array = np.array(img).astype(np.float32) / 255.0

# To tensor
tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).cuda()

# Enhance
with torch.no_grad():
    enhanced = model(tensor)

# Convert back
enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)
```

## Model Details

**Architecture**: Hybrid NightSightNet
- Combines traditional and deep learning approaches
- Optimized for real-time performance
- 172,118 parameters

**Training Configuration:**
- Optimizer: AdamW
- Learning Rate: 1e-4
- Batch Size: 4
- Epochs: 200 (best at 156)
- Loss: Combined (L1 + Perceptual + SSIM)

**Real-time Performance (CUDA GPU):**
- VGA (640×480): 64.6 FPS
- QVGA (320×240): 266 FPS
- HD (1280×720): 13.2 FPS

## Quick Start

```bash
# Single image enhancement
python scripts/enhance_image.py --image path/to/image.jpg --model models/nightsight_best.pth

# Real-time webcam
python scripts/realtime_demo.py --checkpoint models/nightsight_best.pth

# Batch processing
python scripts/enhance_batch.py --input-dir images/ --output-dir enhanced/ --model models/nightsight_best.pth
```

## Citation

If you use this model in your research, please cite:

```
NightSight: Low-Light Image Enhancement Framework
Trained on LOL Dataset (BMVC 2018)
```

## License

See main repository LICENSE file.
