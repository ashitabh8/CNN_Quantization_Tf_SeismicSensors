# Experiment Tracking System

This document describes the new organized experiment tracking system for seismic sensor data classification.

## 🎯 Overview

The experiment tracking system provides:
- **Unique Experiment IDs**: Each training run gets a unique identifier
- **Organized Storage**: Models and logs are stored in structured directories
- **No Overwrites**: Previous experiments are preserved
- **Easy Traceability**: Connect logs to specific models instantly
- **TensorBoard Integration**: Enhanced logging with experiment metadata

## 📁 Directory Structure

```
experiments/
├── 20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4/
│   ├── model/                          # Saved model files
│   │   ├── saved_model.pb
│   │   ├── variables/
│   │   └── assets/
│   ├── logs/                           # TensorBoard logs
│   │   ├── training/                   # Training metrics
│   │   ├── detailed_metrics/           # F1, recall, confusion matrices
│   │   ├── test_metrics/               # Test set evaluation
│   │   └── experiment_metadata/        # Model and config info
│   ├── metadata.json                   # Experiment metadata
│   └── model_summary.txt               # Model architecture summary
├── 20241201_150315_simple_cnn_3classes_32batch_e5f6g7h8/
│   └── ...
```

## 🏷️ Naming Convention

**Format**: `{timestamp}_{model_type}_{num_classes}classes_{batch_size}batch_{config_hash}`

- `timestamp`: YYYYMMDD_HHMMSS
- `model_type`: ultra_lightweight_cnn, simple_cnn, deep_efficient_cnn, etc.
- `num_classes`: Number of output classes
- `batch_size`: Training batch size
- `config_hash`: 8-character hash of key hyperparameters

**Example**: `20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4`

## 🚀 Usage

### Running Training

The training script now automatically creates organized experiments:

```bash
python simple_train_model.py
```

This will:
1. Generate a unique experiment ID
2. Create organized directory structure
3. Save model with consistent naming
4. Log all metrics to TensorBoard with experiment metadata

### Managing Experiments

Use the experiment manager utility:

```bash
# List all experiments
python experiment_manager.py list

# Show experiment summary
python experiment_manager.py summary 20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4

# Get TensorBoard command
python experiment_manager.py tensorboard 20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4

# Compare two experiments
python experiment_manager.py compare exp1_id exp2_id
```

### Viewing Results in TensorBoard

```bash
# View single experiment
tensorboard --logdir experiments/20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4/logs

# Compare multiple experiments
tensorboard --logdir_spec=exp1:experiments/exp1/logs,exp2:experiments/exp2/logs
```

## 📊 TensorBoard Tabs

- **SCALARS**: Training/validation loss and accuracy
- **IMAGES**: Confusion matrices for each epoch
- **HISTOGRAMS**: Model weight distributions
- **TEXT**: Experiment metadata and configuration

## 🔧 Configuration

The system automatically tracks:
- Model architecture and parameters
- Training hyperparameters (batch size, epochs, learning rate)
- Dataset information (sizes, input shape, classes)
- System information (TensorFlow version, Python version)

## 📝 Metadata

Each experiment includes a `metadata.json` file with:

```json
{
  "experiment_id": "20241201_143022_ultra_lightweight_cnn_3classes_32batch_a1b2c3d4",
  "timestamp": "2024-12-01T14:30:22.123456",
  "model_info": {
    "type": "ultra_lightweight_cnn",
    "name": "sequential",
    "input_shape": [2, 7, 256],
    "num_classes": 3,
    "total_params": 12345,
    "trainable_params": 12345
  },
  "training_config": {
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001,
    "input_shape": [2, 7, 256],
    "num_classes": 3,
    "model_type": "ultra_lightweight_cnn",
    "dataset_config": "dataset_config.yaml"
  },
  "dataset_info": {
    "train_size": 1000,
    "val_size": 200,
    "test_size": 300,
    "input_shape": [2, 7, 256],
    "num_classes": 3
  },
  "system_info": {
    "tensorflow_version": "2.x.x",
    "python_version": "3.x.x"
  }
}
```

## 🔄 Migration from Old System

The new system is backward compatible:
- Old training runs will continue to work with fallback logging
- New runs automatically use the organized system
- No changes needed to existing code

## 🎛️ Customization

To customize the experiment tracking:

1. **Modify naming convention** in `experiment_utils.py`:
   ```python
   def generate_experiment_id(model, config, input_shape, num_classes):
       # Customize the ID format here
   ```

2. **Add custom metadata** in `simple_train_model.py`:
   ```python
   dataset_info = {
       'custom_field': 'custom_value',
       # ... existing fields
   }
   ```

3. **Extend directory structure** in `experiment_utils.py`:
   ```python
   subdirs = ['model', 'logs', 'logs/training', 'custom_dir']
   ```

## 🐛 Troubleshooting

**Q: Experiment directory not created?**
A: Check write permissions in the current directory.

**Q: TensorBoard not showing logs?**
A: Ensure the log directory path is correct and contains log files.

**Q: Metadata file missing?**
A: Check if the experiment completed successfully and didn't crash.

**Q: Model not saving?**
A: Verify the model export path and check for disk space.

## 📈 Benefits

✅ **No Overwrites**: Each experiment gets unique directory  
✅ **Easy Comparison**: Side-by-side TensorBoard views  
✅ **Full Traceability**: Metadata links models to logs  
✅ **Scalable**: Works with any number of experiments  
✅ **Organized**: Clean directory structure  
✅ **Reproducible**: All config saved with experiment  
✅ **Backward Compatible**: Works with existing code
