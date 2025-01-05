# Fast PyTorch UNet Training and Inference

A lightweight and fast implementation of UNet for image classification optimized for CPU training and inference.

## Features

- 🚀 Ultra-fast training on CPU
- 💾 Minimal memory usage
- 📦 Simple dataset handling
- ⚡ Efficient inference
- 🔄 Batch processing support

## Installation

```bash
pip install torch torchvision pillow numpy
```

## Directory Structure

```
project/
│
├── data/
│   ├── images/           # Your training images
│   └── annotations.json  # Image annotations
│
├── models/
│   └── saved/           # Saved model checkpoints
│
├── fast_train.py        # Training script
└── fast_inference.py    # Inference script
```

## Data Format

Your `annotations.json` should follow this structure:

```json
[
  {
    "image_name": "image1.jpg",
    "label": 0
  },
  {
    "image_name": "image2.jpg",
    "label": 1
  }
]
```

## Quick Start

### Training

1. Prepare your data:

   - Place images in `data/images/`
   - Create annotations.json in `data/`

2. Run training:

```python
python fast_train.py
```

Key training optimizations:

- 64x64 image size
- Grayscale conversion
- Large batch size (128)
- Aggressive learning rate (0.01)
- Minimal epochs (10)
- Memory-efficient data loading
- Image caching
- Single thread processing

### Inference

1. Single image inference:

```python
from fast_inference import FastInference

inferencer = FastInference('models/saved/model_epoch_10.pth')
prediction = inferencer.predict('path/to/image.jpg')
```

2. Batch inference:

```python
from fast_inference import batch_inference

image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = batch_inference(image_paths, 'models/saved/model_epoch_10.pth')
```

## Configuration

### Training Configuration

Edit these parameters in `fast_train.py`:

```python
# Training hyperparameters
batch_size = 128
num_epochs = 10
learning_rate = 1e-2

# CPU threads
torch.set_num_threads(4)  # Adjust based on your CPU
```

### Inference Configuration

Edit these parameters in `fast_inference.py`:

```python
# Batch inference settings
batch_size = 32  # Adjust based on available RAM
```

## Memory Optimization Tips

1. Training:

   - Reduce image size if needed (32x32 for extremely fast training)
   - Decrease batch size if running out of memory
   - Use grayscale images
   - Enable image caching only if RAM allows

2. Inference:
   - Adjust batch size based on available memory
   - Use torch.no_grad() for inference
   - Clear GPU cache if using CUDA
   - Process images in smaller batches if needed

## Performance Tips

### For Faster Training:

- Increase batch size if memory allows
- Reduce image dimensions
- Skip validation steps
- Reduce model complexity
- Use aggressive learning rates

### For Faster Inference:

- Use batch processing when possible
- Keep model in eval mode
- Disable gradient computation
- Preload and cache transforms
- Use CPU thread optimization

## Troubleshooting

Common issues and solutions:

1. Out of Memory:

   - Reduce batch size
   - Decrease image dimensions
   - Disable image caching

2. Slow Training:

   - Check CPU thread count
   - Increase batch size
   - Reduce image size
   - Simplify model architecture

3. Poor Accuracy:
   - Increase image size
   - Add more training epochs
   - Reduce learning rate
   - Add data augmentation

## Limitations

- Optimized for speed over accuracy
- Limited to grayscale images
- Fixed input size (64x64)
- Minimal model architecture
- CPU-only implementation
