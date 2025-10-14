# Seismic Sensor CNN Training System

A complete training and evaluation pipeline for CNN models on seismic sensor data, with TensorFlow Lite Micro support and comprehensive TensorBoard logging.

## Features

- **CNN Models**: Both 2D CNN (for spectrograms) and 1D CNN (for raw time series)
- **TensorFlow Lite Micro**: Model conversion optimized for microcontrollers
- **TensorBoard Integration**: Comprehensive logging with confusion matrices and metrics
- **Experiment Management**: Unique experiment IDs with organized folder structure
- **Evaluation Metrics**: Accuracy, confusion matrices, classification reports

## File Structure

```
demo_training/
├── models.py                 # CNN model definitions
├── train_and_eval.py         # Training and evaluation pipeline
├── tensorboard_utils.py      # TensorBoard logging utilities
├── example_training.py       # Example usage script
├── demo_dataset_config.yaml  # Dataset configuration
├── dataset.py                # Dataset loading utilities
├── dataset_utils.py          # Dataset helper functions
└── experiments/              # Training experiments (auto-created)
    └── {model_name}_{timestamp}/
        ├── tensorboard/      # TensorBoard logs
        ├── models/           # Saved models
        ├── plots/           # Generated plots
        └── logs/            # Training logs and metrics
```

## Quick Start

### 1. Basic Training

```python
from train_and_eval import SeismicModelTrainer

# Create trainer
trainer = SeismicModelTrainer('demo_dataset_config.yaml', 'MySeismicCNN')

# Train the model
history = trainer.train(epochs=50)

# Evaluate
results = trainer.evaluate()

# Save for TensorFlow Lite Micro
tflite_path, size_info = trainer.save_model_for_tflite_micro()

# Cleanup
trainer.close()
```

### 2. Run Example Script

```bash
python example_training.py
```

### 3. View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir experiments/

# Open browser to http://localhost:6006
```

## Model Architecture

### 2D CNN (for Spectrograms)
- Input: 2D spectrogram data (e.g., 10x11 from Welch processing)
- 3 Conv2D blocks with BatchNorm and Dropout
- Global Average Pooling
- Dense layer with 64 units
- Output: Softmax classification

### 1D CNN (for Raw Time Series)
- Input: 1D time series data (e.g., 200 samples)
- 3 Conv1D blocks with BatchNorm and Dropout
- Global Average Pooling
- Dense layer with 64 units
- Output: Softmax classification

## TensorBoard Logging

The system automatically logs:

- **Scalars**: Loss, accuracy, learning rate
- **Histograms**: Weight and gradient distributions
- **Images**: Confusion matrices
- **Text**: Classification reports, model architecture
- **Graphs**: Model computation graph

## TensorFlow Lite Micro Support

Models are automatically converted to TensorFlow Lite format optimized for microcontrollers:

- **Quantization**: Float16 optimization for size reduction
- **Compatibility**: Supports TensorFlow Lite Micro operations
- **Size Reporting**: Automatic model size calculation

## Experiment Management

Each training run creates a unique experiment with:

- **Unique ID**: `{model_name}_{YYYYMMDD_HHMMSS}`
- **Organized Logs**: Separate directories for different log types
- **Comprehensive Summary**: JSON file with all experiment details

## Configuration

The system uses `demo_dataset_config.yaml` for configuration:

```yaml
vehicle_classification:
    class_names: ["background", "nissan", "lexus", "mazda", "benz"]
    included_classes: ["nissan", "lexus", "mazda", "benz"]
    train_index_file: /path/to/train_index.txt
    val_index_file: /path/to/val_index.txt

max_distance_m: 15
batch_size: 32

spectral_processing:
  method: "welch"  # or "fft" or "none"
  sampling_rate: 100
  welch:
    nperseg: 10
    noverlap: 5
    nfft: 20
    window: "hann"
    scaling: "density"
    detrend: "constant"
```

## API Reference

### SeismicModelTrainer

Main training class with methods:

- `train(epochs=50)`: Train the model
- `evaluate(test_dataset=None)`: Evaluate on validation/test data
- `save_model_for_tflite_micro()`: Convert to TensorFlow Lite
- `save_experiment_summary()`: Save experiment details
- `close()`: Cleanup resources

### TensorBoardLogger

Custom TensorBoard logging with methods:

- `log_scalar(name, value, step)`: Log scalar metrics
- `log_confusion_matrix(y_true, y_pred, step)`: Log confusion matrix
- `log_classification_report(y_true, y_pred, step)`: Log classification report
- `log_model_architecture(model, step)`: Log model summary

## Output Files

### Models
- `full_model.h5`: Complete Keras model
- `model.tflite`: TensorFlow Lite model for microcontrollers
- `best_model.h5`: Best model during training

### Logs
- `training_history.json`: Training metrics over time
- `evaluation_results.json`: Final evaluation metrics
- `model_parameters.json`: Model parameter counts
- `model_size_info.json`: TensorFlow Lite model size
- `experiment_summary.json`: Complete experiment details

## Requirements

```bash
pip install tensorflow tensorflow-model-optimization scikit-learn matplotlib seaborn pyyaml
```

## Usage Examples

### Custom Model Training

```python
# Create custom model
trainer = SeismicModelTrainer('config.yaml', 'CustomCNN')

# Train with custom parameters
history = trainer.train(epochs=100)

# Evaluate and save
results = trainer.evaluate()
tflite_path, size_info = trainer.save_model_for_tflite_micro()
```

### TensorBoard Analysis

```python
# Start TensorBoard
import subprocess
subprocess.run(['tensorboard', '--logdir', 'experiments/'])
```

### Model Loading

```python
import tensorflow as tf

# Load full model
model = tf.keras.models.load_model('experiments/MySeismicCNN_20241201_143022/models/full_model.h5')

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='experiments/MySeismicCNN_20241201_143022/models/model.tflite')
interpreter.allocate_tensors()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Path Issues**: Check configuration file paths
3. **Memory Issues**: Reduce batch size in configuration
4. **TensorBoard Issues**: Check log directory permissions

### Performance Tips

1. **Use GPU**: Ensure TensorFlow GPU support is available
2. **Batch Size**: Optimize batch size for your hardware
3. **Data Pipeline**: Use `tf.data.AUTOTUNE` for optimal performance
4. **Model Size**: Monitor TensorFlow Lite model size for deployment

## Contributing

To extend the system:

1. Add new model architectures in `models.py`
2. Extend TensorBoard logging in `tensorboard_utils.py`
3. Add new evaluation metrics in `train_and_eval.py`
4. Update configuration schema as needed
