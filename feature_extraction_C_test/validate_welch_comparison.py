#!/usr/bin/env python3
"""
Validation script to compare TensorFlow and C implementations of Welch's method.
Generates sine wave test signals and compares PSD outputs.
"""

import numpy as np
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add feature_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_training'))
from feature_utils import tensorflow_welch, WELCH_PARAMS

def generate_sine_wave(length=200, freq=10, sampling_rate=100, amplitude=1.0, phase=0.0):
    """Generate a sine wave signal."""
    t = np.arange(length) / sampling_rate
    return amplitude * np.sin(2 * np.pi * freq * t + phase)

def run_c_implementation(signal, c_executable_path):
    """Run C implementation and return PSD and frequencies."""
    # Save signal to CSV
    np.savetxt('test_signal.csv', signal, fmt='%.15e')
    
    # Run C program
    try:
        result = subprocess.run(['./' + c_executable_path], 
                              capture_output=True, text=True, check=True)
        print("C program output:")
        print(result.stdout)
        if result.stderr:
            print("C program stderr:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running C program: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None, None
    
    # Read C output
    try:
        psd_c = np.loadtxt('test_psd_c.csv')
        freqs_c = np.loadtxt('test_freqs_c.csv')
        return psd_c, freqs_c
    except FileNotFoundError as e:
        print(f"Error reading C output files: {e}")
        return None, None

def compare_psd_outputs(psd_tf, psd_c, freqs_tf, freqs_c, test_name, tolerance=1e-2):
    """Compare PSD outputs and report differences."""
    print(f"\n=== {test_name} ===")
    
    # Check if arrays have same length
    if len(psd_tf) != len(psd_c):
        print(f"ERROR: Length mismatch - TF: {len(psd_tf)}, C: {len(psd_c)}")
        return False
    
    if len(freqs_tf) != len(freqs_c):
        print(f"ERROR: Frequency length mismatch - TF: {len(freqs_tf)}, C: {len(freqs_c)}")
        return False
    
    # Compare frequencies (should be very close)
    freq_diff = np.abs(freqs_tf - freqs_c)
    max_freq_diff = np.max(freq_diff)
    print(f"Max frequency difference: {max_freq_diff:.2e}")
    
    # Compare PSD values
    psd_diff = np.abs(psd_tf - psd_c)
    max_diff = np.max(psd_diff)
    mean_abs_error = np.mean(psd_diff)
    relative_error = np.mean(psd_diff / (np.abs(psd_tf) + 1e-12))
    
    print(f"Max PSD difference: {max_diff:.6e}")
    print(f"Mean absolute error: {mean_abs_error:.6e}")
    print(f"Mean relative error: {relative_error:.6e}")
    
    # Check if within tolerance
    success = max_diff < tolerance
    print(f"Within tolerance ({tolerance:.0e}): {'✓ PASS' if success else '✗ FAIL'}")
    
    if not success:
        # Find the worst differences
        worst_indices = np.argsort(psd_diff)[-5:]
        print("Worst differences:")
        for idx in worst_indices:
            print(f"  Index {idx}: TF={psd_tf[idx]:.6e}, C={psd_c[idx]:.6e}, diff={psd_diff[idx]:.6e}")
    
    return success

def compile_c_program():
    """Compile the C test program."""
    c_file = "test_welch_comparison.c"
    executable = "test_welch_comparison"
    
    if not os.path.exists(c_file):
        print(f"Error: {c_file} not found")
        return None
    
    try:
        # Compile with math library
        result = subprocess.run(['gcc', '-o', executable, c_file, '-lm'], 
                              capture_output=True, text=True, check=True)
        print("C program compiled successfully")
        return executable
    except subprocess.CalledProcessError as e:
        print(f"Error compiling C program: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

def main():
    """Main validation function."""
    print("Welch's Method Implementation Comparison")
    print("=======================================")
    print(f"TensorFlow parameters: {WELCH_PARAMS}")
    print()
    
    # Compile C program
    c_executable = compile_c_program()
    if c_executable is None:
        print("Failed to compile C program. Exiting.")
        return
    
    # Test cases with different frequencies
    test_cases = [
        {'freq': 10, 'name': '10Hz sine wave'},
        {'freq': 25, 'name': '25Hz sine wave'},
        {'freq': 35, 'name': '35Hz sine wave'},
        {'freq': 40, 'name': '40Hz sine wave'},
        {'freq': 5, 'name': '5Hz sine wave (low frequency)'},
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"{'='*60}")
        
        # Generate test signal
        signal = generate_sine_wave(
            length=200, 
            freq=test['freq'], 
            sampling_rate=WELCH_PARAMS['sampling_rate']
        )
        
        print(f"Signal: {len(signal)} samples, {test['freq']} Hz")
        print(f"Signal range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
        
        # Compute PSD using TensorFlow
        print("\nComputing PSD with TensorFlow...")
        psd_tf = tensorflow_welch(signal, **WELCH_PARAMS)
        freqs_tf = np.linspace(0, WELCH_PARAMS['sampling_rate']/2, len(psd_tf))
        
        print(f"TensorFlow PSD: {len(psd_tf)} bins")
        print(f"Frequency range: {freqs_tf[0]:.2f} - {freqs_tf[-1]:.2f} Hz")
        print(f"PSD range: [{np.min(psd_tf):.6e}, {np.max(psd_tf):.6e}]")
        
        # Run C implementation
        print("\nComputing PSD with C implementation...")
        psd_c, freqs_c = run_c_implementation(signal, c_executable)
        
        if psd_c is None or freqs_c is None:
            print("Failed to get C results")
            all_passed = False
            continue
        
        print(f"C PSD: {len(psd_c)} bins")
        print(f"Frequency range: {freqs_c[0]:.2f} - {freqs_c[-1]:.2f} Hz")
        print(f"PSD range: [{np.min(psd_c):.6e}, {np.max(psd_c):.6e}]")
        
        # Compare results
        test_passed = compare_psd_outputs(psd_tf, psd_c, freqs_tf, freqs_c, test['name'])
        all_passed = all_passed and test_passed
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED - Implementations match within tolerance!")
    else:
        print("✗ SOME TESTS FAILED - Implementations differ significantly")
    
    # Cleanup
    cleanup_files = ['test_signal.csv', 'test_psd_c.csv', 'test_freqs_c.csv', 'test_welch_comparison']
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("\nCleanup completed.")

if __name__ == "__main__":
    main()
