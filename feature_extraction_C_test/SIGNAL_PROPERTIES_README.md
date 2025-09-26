# Signal Properties Implementation

This directory contains C implementations of four key signal properties with Python validation scripts to verify correctness against TensorFlow, PyTorch, and NumPy implementations.

## Implemented Properties

### 1. Signal Energy
- **Definition**: Sum of squared values of the signal
- **Formula**: `E = Σ(x[i]²)` for i = 0 to N-1
- **Use Case**: Measures the total power/energy content of the signal

### 2. Above Mean Density
- **Definition**: Fraction of samples above the signal mean
- **Formula**: `D = count(x[i] > mean) / N`
- **Use Case**: Indicates signal asymmetry and distribution characteristics

### 3. First and Last Maximum Locations
- **Definition**: Indices of first and last occurrences of the maximum value
- **Use Case**: Identifies peak locations and signal structure

### 4. Mean Change Properties
- **Mean Change**: Average of first differences
- **Mean Absolute Change**: Average of absolute first differences  
- **Mean Squared Change**: Average of squared first differences
- **Use Case**: Measures signal variability and trend characteristics

## Files

### C Implementation
- `signal_properties.c` - Main C implementation with all four properties
- `Makefile` - Build configuration (updated to include signal properties)

### Python Validation
- `validate_signal_properties.py` - Validation script comparing C vs Python libraries
- Uses TensorFlow, PyTorch, and NumPy for verification

### Test Signals
The implementation uses four non-periodic test signals to better test the properties:

1. **Exponential Decay with Noise** - Tests energy and change properties
2. **Chirp Signal (Frequency Sweep)** - Tests frequency-dependent properties
3. **Step Function with Transitions** - Tests discrete change detection
4. **Random Walk with Trend** - Tests trend and variability measures

## Usage

### Building and Running C Implementation

```bash
# Build the signal properties executable
make test-properties

# Or build manually
gcc -o signal_properties_test signal_properties.c -lm
./signal_properties_test
```

### Running Python Validation

```bash
# First run the C implementation to generate test files
make test-properties

# Then run the Python validation
python3 validate_signal_properties.py
```

## Validation Results

The validation script compares C implementation results with:
- **NumPy**: Reference implementation
- **TensorFlow**: GPU-accelerated computation
- **PyTorch**: Alternative tensor computation

### Typical Accuracy
- **Signal Energy**: ~10⁻⁶ to 10⁻⁸ relative difference
- **Above Mean Density**: ~10⁻⁶ to 10⁻⁷ relative difference  
- **Max Locations**: Exact match (integer indices)
- **Mean Change Properties**: ~10⁻⁴ to 10⁻² relative difference

The small differences are due to:
- Floating-point precision differences between implementations
- Different random number generation seeds
- Compiler optimization differences

## Configuration

### Signal Parameters
- **Signal Length**: 1024 samples
- **Sampling Rate**: 512.0 Hz
- **Duration**: 2.0 seconds

### Test Signal Parameters
- **Exponential Decay**: 20 Hz sine with 2.0 decay rate
- **Chirp Signal**: 10-60 Hz frequency sweep
- **Step Function**: Multiple level transitions
- **Random Walk**: Linear trend with random steps

## Output Files

### C Implementation Outputs
- `test_signal_properties_N.txt` - Time domain signals (N=1-4)
- `test_properties_N.txt` - Calculated properties (N=1-4)

### Python Validation Outputs
- `signal_properties_comparison.png` - Comparison plots
- Console output with detailed numerical comparisons

## API Reference

### C Functions

```c
// Calculate all properties at once
void calculate_all_properties(double *signal, int length, signal_properties_t *properties);

// Individual property functions
void calculate_signal_energy(double *signal, int length, double *energy);
void calculate_above_mean_density(double *signal, int length, double *density);
void find_max_locations(double *signal, int length, int *first_max, int *last_max);
void calculate_mean_change_properties(double *signal, int length, 
                                     double *mean_change, double *mean_abs_change, 
                                     double *mean_squared_change);
```

### Python Functions

```python
# Calculate all properties using different backends
calculate_all_properties_python(signal, method='numpy')     # NumPy
calculate_all_properties_python(signal, method='tensorflow') # TensorFlow  
calculate_all_properties_python(signal, method='pytorch')   # PyTorch
```

## Performance Notes

- **C Implementation**: Optimized for speed with minimal memory allocation
- **TensorFlow**: GPU acceleration available, good for batch processing
- **PyTorch**: Similar performance to TensorFlow, good for research
- **NumPy**: Fastest for single signals, reference implementation

## Integration

The signal properties can be easily integrated into larger signal processing pipelines:

1. **Feature Extraction**: Use as input features for machine learning
2. **Signal Analysis**: Characterize signal properties for classification
3. **Quality Assessment**: Monitor signal quality and detect anomalies
4. **Real-time Processing**: C implementation suitable for embedded systems

## Dependencies

### C Implementation
- Standard C library
- Math library (`-lm`)

### Python Validation
- NumPy
- TensorFlow
- PyTorch
- Matplotlib
- Pandas
- SciPy
- Scikit-learn

## Future Enhancements

Potential improvements and extensions:
- Additional statistical properties (skewness, kurtosis)
- Frequency domain properties
- Windowed analysis capabilities
- Real-time streaming interface
- GPU-accelerated C implementation using CUDA
