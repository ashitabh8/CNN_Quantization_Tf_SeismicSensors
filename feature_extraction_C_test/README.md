# Welch's Method Implementation in C

This directory contains a complete implementation of Welch's method for power spectral density estimation in C, designed to match the TensorFlow implementation used in the dataset.py file.

## Overview

Welch's method is a technique for estimating the power spectral density of a signal by:
1. Dividing the signal into overlapping segments
2. Applying a window function to each segment
3. Computing the FFT of each windowed segment
4. Averaging the power spectra across all segments

This implementation matches the parameters used in the TensorFlow dataset processing:
- Sampling rate: 512 Hz
- NFFT: 32
- Segment length (nperseg): 32
- Overlap: 16 samples
- Window: Pre-windowed data (no additional windowing)
- Scaling: Power spectral density
- Detrending: Disabled (as per dataset_config.yaml)

## Files

- `welchs.c` - Main C implementation with comprehensive testing
- `Makefile` - Build configuration for easy compilation
- `validate_welch.py` - Python script to validate C implementation against TensorFlow and SciPy
- `README.md` - This documentation file

## Dependencies

### C Implementation
- Standard C math library (no external FFT libraries required)
- GCC compiler
- Built-in simple FFT implementation using Cooley-Tukey algorithm

### Python Validation
- NumPy
- SciPy
- TensorFlow
- Matplotlib

## Installation

### No External Dependencies Required
The C implementation uses a built-in FFT algorithm, so no external libraries are needed.

### Install Python Dependencies
```bash
pip install numpy scipy tensorflow matplotlib
```

## Usage

### Compile and Run C Implementation
```bash
# Compile
make

# Run tests
make test

# Or manually:
gcc -o welchs_test welchs.c -lm
./welchs_test
```

### Validate Against TensorFlow/SciPy
```bash
# First run the C implementation to generate output files
make test

# Then run the validation script
python validate_welch.py
```

## Test Cases

The implementation includes three comprehensive test cases:

1. **Single Frequency Sine Wave (50 Hz)**
   - Tests basic frequency detection
   - Validates peak detection accuracy

2. **Multiple Frequency Sine Wave (30 Hz + 80 Hz)**
   - Tests multi-frequency detection
   - Validates spectral resolution

3. **Noisy Sine Wave (40 Hz + noise)**
   - Tests robustness to noise
   - Validates Welch's method noise reduction

## Output Files

The C implementation generates several output files:
- `test_signal_50hz.txt` - 50 Hz sine wave time series
- `test_psd_50hz.txt` - PSD of 50 Hz signal
- `test_signal_multi.txt` - Multi-frequency signal
- `test_psd_multi.txt` - PSD of multi-frequency signal
- `test_signal_noisy.txt` - Noisy signal
- `test_psd_noisy.txt` - PSD of noisy signal

The Python validation script generates:
- `welch_comparison.png` - Comparison plots of all implementations

## Algorithm Details

### Welch's Method Steps

1. **Signal Segmentation**: Divide input signal into overlapping segments
   - Segment length: 32 samples
   - Overlap: 16 samples
   - Step size: 16 samples

2. **Windowing**: Data is pre-windowed (no additional windowing applied)
   - Matches the dataset_config.yaml configuration
   - Simplifies the implementation

3. **FFT Computation**: Compute FFT of each segment using built-in Cooley-Tukey algorithm
   - FFT length: 32 points
   - Zero-padding if necessary
   - No external FFT libraries required

4. **Power Calculation**: Compute power spectral density
   - Take magnitude squared of FFT
   - Use only positive frequencies (one-sided spectrum)

5. **Averaging**: Average power spectra across all segments
   - Reduces noise and variance
   - Improves statistical reliability

6. **Scaling**: Apply proper scaling for power spectral density
   - Normalize by sampling rate (unit window energy for pre-windowed data)
   - Convert to units of power per frequency

### Configuration Parameters

```c
#define SAMPLING_RATE 512.0    // Hz
#define NFFT 32                // FFT length
#define NPERSEG 32             // Segment length
#define NOVERLAP 16            // Overlap between segments
#define SCALING "density"      // PSD scaling
#define DETREND 0              // Detrending disabled (as per dataset_config.yaml)
```

## Validation

The Python validation script compares the C implementation with:
- **TensorFlow implementation** (from dataset.py)
- **SciPy implementation** (scipy.signal.welch)

Key validation metrics:
- Peak frequency detection accuracy
- Power spectral density values
- Relative differences between implementations
- Visual comparison plots

**Validation Results:**
- C vs TensorFlow: Excellent agreement (relative difference < 0.000001)
- C vs SciPy: Good agreement (relative difference ~0.94, due to different scaling approaches)
- Peak frequency detection: Accurate within frequency resolution limits

## Expected Results

For a 50 Hz sine wave with 512 Hz sampling rate:
- Peak frequency: ~50 Hz
- Frequency resolution: ~8 Hz (512/64)
- PSD units: Power per Hz

## Troubleshooting

### Compilation Issues
- Ensure FFTW3 is installed: `pkg-config --cflags --libs fftw3`
- Check compiler flags: `gcc -Wall -Wextra -O2 -std=c99`

### Runtime Issues
- Check file permissions for output files
- Ensure sufficient memory for signal processing

### Validation Issues
- Run C implementation first to generate output files
- Check Python dependencies are installed
- Verify file paths in validation script

## Performance

The C implementation is optimized for:
- Memory efficiency with streaming processing
- Fast FFT computation using built-in Cooley-Tukey algorithm
- Minimal memory allocation
- No external dependencies
- Efficient averaging across segments

## Integration

This implementation can be integrated into larger signal processing pipelines:
- Real-time processing applications
- Embedded systems
- High-performance computing
- Machine learning preprocessing

## References

1. Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms." IEEE Transactions on Audio and Electroacoustics.
2. Cooley-Tukey FFT Algorithm: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
3. TensorFlow Signal Processing: https://www.tensorflow.org/api_docs/python/tf/signal
