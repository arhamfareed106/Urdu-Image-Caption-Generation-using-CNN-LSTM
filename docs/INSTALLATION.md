# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: Intel i5 or AMD equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- **Storage**: 10GB free space

### Software Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/urdu-caption-generation.git
cd urdu-caption-generation
```

### 2. Set Up Python Environment

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n urdu-caption python=3.8
conda activate urdu-caption
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Install Additional Dependencies

```bash
# For Urdu text processing
pip install arabic-reshaper python-bidi

# For Jupyter notebooks
pip install jupyter ipykernel

# For GPU support (if available)
pip install tensorflow-gpu
```

### 5. Verify Installation

```bash
# Test TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Test Urdu text processing
python -c "import arabic_reshaper; print('Urdu processing ready')"
```

## GPU Setup (Optional)

### NVIDIA GPU Setup

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install appropriate driver for your GPU

2. **Install CUDA Toolkit**
   ```bash
   # Download and install CUDA 11.0+ from NVIDIA website
   # Verify installation
   nvcc --version
   ```

3. **Install cuDNN**
   - Download cuDNN from NVIDIA Developer website
   - Extract and copy files to CUDA installation directory

4. **Install TensorFlow GPU**
   ```bash
   pip install tensorflow-gpu
   ```

### Verify GPU Support

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# If you encounter SSL errors
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org tensorflow

# For specific TensorFlow version
pip install tensorflow==2.8.0
```

#### 2. Memory Issues
```python
# Limit GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 3. Urdu Font Issues
```python
# Install system fonts for Urdu
# On Ubuntu:
sudo apt-get install fonts-noto-core fonts-noto-ui-core

# On Windows:
# Download and install Jameel Noori Nastaleeq font
```

#### 4. Jupyter Kernel Issues
```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=urdu-caption --display-name="Urdu Caption Generation"
```

### Environment Variables

Add these to your environment for optimal performance:

```bash
# On Windows (PowerShell)
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
$env:CUDA_VISIBLE_DEVICES = "0"

# On Linux/macOS
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0
```

## Quick Test

After installation, run this quick test:

```python
# test_installation.py
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display

print("✅ All dependencies installed successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Test Urdu text processing
urdu_text = "سلام دنیا"
reshaped = arabic_reshaper.reshape(urdu_text)
bidi_text = get_display(reshaped)
print(f"Urdu text processing: {bidi_text}")
```

Run the test:
```bash
python test_installation.py
```

## Next Steps

After successful installation:

1. **Download the dataset** (see [Dataset Setup](DATASET.md))
2. **Run the training script** (see [Training Guide](TRAINING.md))
3. **Test the model** (see [Usage Guide](USAGE.md))

## Support

If you encounter any issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/urdu-caption-generation/issues)
3. Create a new issue with detailed error information 