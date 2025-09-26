#!/usr/bin/env python3
"""
Validation script to compare C implementation of Welch's method with TensorFlow implementation.
This script reads the output files from the C implementation and compares them with
equivalent TensorFlow calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.signal
import tensorflow as tf

# Configuration matching the C implementation and dataset_config.yaml
SAMPLING_RATE = 512.0
NFFT = 32
NPERSEG = 32
NOVERLAP = 16
SCALING = 'density'
DETREND = False  # Disabled as per dataset_config.yaml

def load_c_results(filename):
    """Load results from C implementation output file."""
    try:
        data = np.loadtxt(filename, skiprows=1)  # Skip header
        if data.ndim == 1:
            # Single column case
            return data
        else:
            # Two column case (frequency, psd)
            return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def generate_test_signals():
    """Generate the same test signals as the C implementation."""
    signal_length = 1024
    t = np.arange(signal_length) / SAMPLING_RATE
    
    # Test 1: Single frequency sine wave (50 Hz)
    signal1 = np.sin(2 * np.pi * 50.0 * t)
    
    # Test 2: Multiple frequency sine wave (30 Hz + 80 Hz)
    signal2 = np.sin(2 * np.pi * 30.0 * t) + 0.5 * np.sin(2 * np.pi * 80.0 * t)
    
    # Test 3: Noisy sine wave (40 Hz + noise)
    np.random.seed(42)  # Same seed as C implementation
    noise = (np.random.random(signal_length) - 0.5) * 0.2
    signal3 = np.sin(2 * np.pi * 40.0 * t) + noise
    
    return signal1, signal2, signal3, t

def tensorflow_welch(signal, sampling_rate=SAMPLING_RATE, nperseg=NPERSEG, 
                    noverlap=NOVERLAP, nfft=NFFT, scaling=SCALING, detrend=DETREND):
    """
    Apply Welch's method using TensorFlow (matching the dataset.py implementation).
    """
    # Convert to TensorFlow tensor
    signal_tf = tf.convert_to_tensor(signal, dtype=tf.float32)
    
    # Apply detrending if specified
    if detrend == 'constant':
        signal_tf = signal_tf - tf.reduce_mean(signal_tf)
    elif detrend == 'linear':
        signal_tf = tf.convert_to_tensor(signal - signal.detrend('linear'), dtype=tf.float32)
    
    # No windowing applied (data is pre-windowed as per dataset_config.yaml)
    window_tf = tf.ones(nperseg, dtype=tf.float32)
    
    # Calculate number of segments
    step = nperseg - noverlap
    nsegments = (len(signal) - noverlap) // step
    
    # Initialize PSD accumulator
    psd_accumulator = tf.zeros(nfft // 2 + 1, dtype=tf.float32)
    
    # Process each segment
    for seg in range(nsegments):
        start_idx = seg * step
        end_idx = start_idx + nperseg
        
        # Extract segment
        segment = signal_tf[start_idx:end_idx]
        
        # Apply window
        segment_windowed = segment * window_tf
        
        # Zero-pad if necessary
        if nfft > nperseg:
            padding = nfft - nperseg
            segment_padded = tf.pad(segment_windowed, [[0, padding]])
        else:
            segment_padded = segment_windowed[:nfft]
        
        # Apply FFT
        fft_result = tf.signal.fft(tf.cast(segment_padded, tf.complex64))
        
        # Take only positive frequencies
        fft_positive = fft_result[:nfft//2+1]
        
        # Compute power spectral density
        psd_segment = tf.abs(fft_positive) ** 2
        
        # Accumulate
        psd_accumulator += psd_segment
    
    # Average across segments
    psd_avg = psd_accumulator / nsegments
    
    # Apply scaling
    if scaling == 'density':
        # Convert to power spectral density
        # Since data is pre-windowed, we assume unit window energy
        psd_avg = psd_avg / sampling_rate
    
    return psd_avg.numpy()

def scipy_welch(signal, sampling_rate=SAMPLING_RATE, nperseg=NPERSEG, 
               noverlap=NOVERLAP, nfft=NFFT, scaling=SCALING, detrend=DETREND):
    """Apply Welch's method using scipy.signal.welch for comparison."""
    detrend_type = 'constant' if detrend else False
    scaling_type = 'density' if scaling == 'density' else 'spectrum'
    
    freqs, psd = scipy.signal.welch(signal, fs=sampling_rate, nperseg=nperseg, 
                                   noverlap=noverlap, nfft=nfft, window='boxcar',  # No windowing
                                   detrend=detrend_type, scaling=scaling_type)
    
    return freqs, psd

def compare_implementations():
    """Compare C, TensorFlow, and SciPy implementations."""
    print("=== Welch's Method Implementation Comparison ===\n")
    
    # Generate test signals
    signal1, signal2, signal3, t = generate_test_signals()
    
    # Test cases
    test_cases = [
        ("Single frequency (50 Hz)", signal1, "test_psd_50hz.txt"),
        ("Multiple frequencies (30 Hz + 80 Hz)", signal2, "test_psd_multi.txt"),
        ("Noisy signal (40 Hz + noise)", signal3, "test_psd_noisy.txt")
    ]
    
    for test_name, test_signal, c_output_file in test_cases:
        print(f"Testing: {test_name}")
        print("-" * 50)
        
        # Check if C output file exists
        if not os.path.exists(c_output_file):
            print(f"Warning: C output file {c_output_file} not found. Run the C implementation first.")
            continue
        
        # Load C results
        c_freqs, c_psd = load_c_results(c_output_file)
        if c_freqs is None:
            continue
        
        # Calculate TensorFlow results
        tf_psd = tensorflow_welch(test_signal)
        tf_freqs = np.linspace(0, SAMPLING_RATE/2, len(tf_psd))
        
        # Calculate SciPy results
        sp_freqs, sp_psd = scipy_welch(test_signal)
        
        # Compare results
        print(f"Signal length: {len(test_signal)}")
        print(f"Number of segments: {(len(test_signal) - NOVERLAP) // (NPERSEG - NOVERLAP)}")
        print(f"PSD length - C: {len(c_psd)}, TF: {len(tf_psd)}, SciPy: {len(sp_psd)}")
        
        # Find peak frequencies
        c_peak_idx = np.argmax(c_psd)
        tf_peak_idx = np.argmax(tf_psd)
        sp_peak_idx = np.argmax(sp_psd)
        
        print(f"Peak frequency - C: {c_freqs[c_peak_idx]:.2f} Hz, TF: {tf_freqs[tf_peak_idx]:.2f} Hz, SciPy: {sp_freqs[sp_peak_idx]:.2f} Hz")
        print(f"Peak PSD - C: {c_psd[c_peak_idx]:.6e}, TF: {tf_psd[tf_peak_idx]:.6e}, SciPy: {sp_psd[sp_peak_idx]:.6e}")
        
        # Calculate relative differences
        if len(c_psd) == len(tf_psd):
            rel_diff_tf = np.mean(np.abs(c_psd - tf_psd) / (c_psd + 1e-10))
            print(f"Relative difference (C vs TF): {rel_diff_tf:.6f}")
        
        if len(c_psd) == len(sp_psd):
            rel_diff_sp = np.mean(np.abs(c_psd - sp_psd) / (c_psd + 1e-10))
            print(f"Relative difference (C vs SciPy): {rel_diff_sp:.6f}")
        
        print()
    
    # Create comparison plots
    create_comparison_plots(test_cases)

def create_comparison_plots(test_cases):
    """Create comparison plots for all test cases."""
    print("Creating comparison plots...")
    
    # Generate test signals
    signal1, signal2, signal3, t = generate_test_signals()
    signals = [signal1, signal2, signal3]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Welch's Method Implementation Comparison", fontsize=16)
    
    for i, (test_name, test_signal, c_output_file) in enumerate(test_cases):
        if not os.path.exists(c_output_file):
            continue
        
        # Load C results
        c_freqs, c_psd = load_c_results(c_output_file)
        if c_freqs is None:
            continue
        
        # Calculate other implementations
        tf_psd = tensorflow_welch(test_signal)
        tf_freqs = np.linspace(0, SAMPLING_RATE/2, len(tf_psd))
        sp_freqs, sp_psd = scipy_welch(test_signal)
        
        # Plot time domain
        axes[i, 0].plot(t, test_signal)
        axes[i, 0].set_title(f"{test_name} - Time Domain")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)
        
        # Plot frequency domain comparison
        axes[i, 1].plot(c_freqs, c_psd, 'b-', label='C Implementation', linewidth=2)
        axes[i, 1].plot(tf_freqs, tf_psd, 'r--', label='TensorFlow', linewidth=2)
        axes[i, 1].plot(sp_freqs, sp_psd, 'g:', label='SciPy', linewidth=2)
        axes[i, 1].set_title(f"{test_name} - PSD Comparison")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("PSD")
        axes[i, 1].legend()
        axes[i, 1].grid(True)
        axes[i, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('welch_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'welch_comparison.png'")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

def main():
    """Main function."""
    print("Welch's Method Validation Script")
    print("===============================\n")
    
    # Check if required files exist
    required_files = ["test_psd_50hz.txt", "test_psd_multi.txt", "test_psd_noisy.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing C implementation output files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run the C implementation first:")
        print("  make test")
        print("\nOr manually:")
        print("  gcc -o welchs_test welchs.c -lfftw3 -lm")
        print("  ./welchs_test")
        return 1
    
    # Run comparison
    compare_implementations()
    
    print("Validation completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
