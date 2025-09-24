# Environment Setup Instructions

This document provides instructions for recreating the conda environment used in this CNN Quantization TensorFlow Seismic Sensors project.

## Files Provided

- `environment.yml` - Conda environment file (recommended)
- `requirements.txt` - Pip requirements file (alternative)

## Option 1: Using Conda (Recommended)

### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### Steps
1. Navigate to the project directory:
   ```bash
   cd /path/to/CNN_Quantization_Tf_SeismicSensors
   ```

2. Create the conda environment from the environment file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate cnn_quantization_tf_seismic
   ```

4. Verify the installation:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## Option 2: Using Pip (Alternative)

### Prerequisites
- Python 3.13.2 (or compatible version)
- pip package manager

### Steps
1. Create a virtual environment (recommended):
   ```bash
   python -m venv cnn_quantization_env
   source cnn_quantization_env/bin/activate  # On Windows: cnn_quantization_env\Scripts\activate
   ```

2. Install packages from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## Environment Details

- **Python Version**: 3.13.2
- **TensorFlow**: 2.20.0 (with GPU support)
- **PyTorch**: 2.6.0 (with CUDA 12.9 support)
- **Key Libraries**: 
  - Keras 3.11.3
  - NumPy 1.26.4
  - Pandas 2.2.3
  - Scikit-learn 1.6.1
  - Matplotlib 3.10.1
  - Seaborn 0.13.2
  - Hugging Face Transformers 4.49.0

## GPU Support

This environment includes NVIDIA CUDA 12.9 libraries for GPU acceleration. Make sure you have:
- NVIDIA GPU with CUDA support
- Compatible NVIDIA drivers installed
- CUDA toolkit 12.9 (optional, libraries are included in the environment)

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: If you encounter CUDA-related errors, you can install CPU-only versions:
   ```bash
   pip install tensorflow-cpu==2.20.0
   pip install torch==2.6.0+cpu torchvision==0.21.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Version Conflicts**: If you encounter version conflicts, try creating a fresh environment:
   ```bash
   conda env remove -n cnn_quantization_tf_seismic
   conda env create -f environment.yml
   ```

3. **Memory Issues**: Some packages require significant memory during installation. Consider:
   - Closing other applications
   - Using `--no-cache-dir` flag with pip
   - Installing packages one by one if needed

### Getting Help

If you encounter issues not covered here:
1. Check the [TensorFlow installation guide](https://www.tensorflow.org/install)
2. Check the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
3. Verify your Python and system compatibility

## Notes

- The environment includes both TensorFlow and PyTorch for maximum flexibility
- All major ML libraries are included for seismic sensor data processing
- The environment is optimized for CNN quantization research
- Some packages may take several minutes to install due to their size
