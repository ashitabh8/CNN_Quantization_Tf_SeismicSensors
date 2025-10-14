# SeismicDataset - 2D Spectrogram Implementation

## Overview
The `SeismicDataset` class creates 2D spectrograms from seismic sensor data for vehicle classification using TensorFlow.

## Data Structure
- **Input**: Seismic data with shape `(1, 10, 20)`
  - `1` = single channel
  - `10` = 10 time steps over 2 seconds
  - `20` = 20 data points per time step
  - **Total**: 200 samples over 2 seconds at 100Hz sampling rate
- **Output**: 2D spectrogram with shape `(10, 11)`
  - `10` = time dimension (10 time steps)
  - `11` = frequency dimension (nfft//2+1 frequency bins)
  - Single channel dimension is squeezed out

## Configuration Parameters

### Welch Method Parameters (demo_dataset_config.yaml)
```yaml
spectral_processing:
  method: "welch"  # Creates 2D spectrogram
  sampling_rate: 100  # 100Hz (100 samples per second)
  
  welch:
    nperseg: 10     # Use all 10 time points as one segment
    noverlap: 5     # 50% overlap between segments
    nfft: 20        # FFT length - matches 20 data points per time step
    window: "hann"  # Hann window for better spectral estimation
    scaling: "density"  # Power spectral density
    detrend: "constant"  # Remove DC component
```

## Key Features

### 1. **2D Spectrogram Creation**
- Converts time series data to frequency-time representation
- Preserves temporal structure while extracting frequency features
- More suitable for CNN-based classification than flattened data

### 2. **Proper Data Flow**
```
Raw Data (1,10,20) → Welch Processing → 2D Spectrogram (1,10,11)
```
- No unnecessary flattening when Welch preprocessing is applied
- Maintains 2D structure for better feature learning

### 3. **Batch Processing**
- Batch shape: `(batch_size, 10, 11)`
- Lazy loading with TensorFlow Dataset
- Memory-efficient processing
- Redundant single channel dimension is squeezed out

## Usage Example

```python
# Create dataset with Welch preprocessing
train_dataset = SeismicDataset(
    train_mapping=train_mapping,
    val_mapping=val_mapping,
    task="vehicle_classification",
    spectral_processing=config.get('spectral_processing', {}),
    is_training=True
)

# Convert to TensorFlow dataset
train_tf_dataset = train_dataset.to_tf_dataset(
    batch_size=32,
    shuffle_buffer_size=1000,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Use in training
for batch_data, batch_labels in train_tf_dataset:
    # batch_data shape: (32, 10, 11) - 2D spectrogram
    # batch_labels shape: (32, 5) - one-hot encoded
    pass
```

## Benefits of 2D Spectrogram Approach

1. **Temporal-Frequency Representation**: Captures both time and frequency patterns
2. **CNN-Friendly**: 2D structure is ideal for convolutional neural networks
3. **Feature Rich**: Frequency domain reveals patterns not visible in time domain
4. **Robust**: Welch method reduces noise through averaging
5. **Interpretable**: Spectrograms are visually interpretable

## Comparison with Previous Approach

| Aspect | Previous (Flattened) | Current (2D Spectrogram) |
|--------|----------------------|---------------------------|
| Shape | `(batch_size, 200)` | `(batch_size, 10, 11)` |
| Structure | 1D flattened | 2D time-frequency |
| CNN Suitability | Poor | Excellent |
| Feature Learning | Limited | Rich |
| Interpretability | Low | High |

## Technical Details

### Welch Method Implementation
- Applies FFT to each time step (20 data points)
- Takes positive frequencies only (nfft//2+1 = 11 bins)
- Computes power spectral density
- Maintains temporal dimension (10 time steps)

### Memory Efficiency
- Lazy loading with `tf.data.Dataset`
- On-demand processing
- Parallel data loading
- Automatic prefetching

This implementation provides a much more appropriate representation for seismic vehicle classification using deep learning approaches.
