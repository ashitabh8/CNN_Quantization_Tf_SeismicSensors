#!/usr/bin/env python3
"""
Validation script to compare C implementation of signal properties with Python library implementations.
This script reads the output files from the C implementation and compares them with
equivalent TensorFlow, NumPy, and SciPy calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import tensorflow as tf
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configuration matching the C implementation
SIGNAL_LENGTH = 1024
SAMPLING_RATE = 512.0

def load_c_properties(filename):
    """Load properties from C implementation output file."""
    try:
        df = pd.read_csv(filename)
        properties = {}
        for _, row in df.iterrows():
            property_name = row['Property']
            value = row['Value']
            # Convert numeric values
            try:
                properties[property_name] = float(value)
            except ValueError:
                properties[property_name] = int(value)
        return properties
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def load_c_signal(filename):
    """Load signal from C implementation output file."""
    try:
        df = pd.read_csv(filename)
        return df['Amplitude'].values
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def generate_test_signals():
    """Generate the same test signals as the C implementation."""
    t = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE
    
    signals = {}
    
    # Test 1: Exponential decay with noise
    np.random.seed(42)
    decay = np.exp(-t * 2.0)
    noise = (np.random.random(SIGNAL_LENGTH) - 0.5) * 0.1
    signals[1] = decay * np.sin(2.0 * np.pi * 20.0 * t) + noise
    
    # Test 2: Chirp signal (frequency sweep)
    freq = 10.0 + 50.0 * t  # Frequency sweep from 10 to 60 Hz
    signals[2] = np.sin(2.0 * np.pi * freq * t) * np.exp(-t * 0.5)
    
    # Test 3: Step function with transitions
    np.random.seed(42)
    step_signal = np.zeros(SIGNAL_LENGTH)
    step_signal[t < 0.5] = 1.0
    step_signal[(t >= 0.5) & (t < 1.0)] = -0.5
    step_signal[(t >= 1.0) & (t < 1.5)] = 0.8
    step_signal[t >= 1.5] = 0.0
    # Add noise
    noise = (np.random.random(SIGNAL_LENGTH) - 0.5) * 0.05
    signals[3] = step_signal + noise
    
    # Test 4: Random walk with trend
    np.random.seed(123)
    random_walk = np.zeros(SIGNAL_LENGTH)
    trend = 0.1 * t  # Linear trend
    for i in range(1, SIGNAL_LENGTH):
        random_step = (np.random.random() - 0.5) * 0.2
        random_walk[i] = random_walk[i-1] + random_step + trend[i]
    signals[4] = random_walk
    
    return signals, t

def numpy_signal_energy(signal):
    """Calculate signal energy using NumPy."""
    return np.sum(signal ** 2)

def tensorflow_signal_energy(signal):
    """Calculate signal energy using TensorFlow."""
    signal_tf = tf.convert_to_tensor(signal, dtype=tf.float32)
    return tf.reduce_sum(signal_tf ** 2).numpy()

def pytorch_signal_energy(signal):
    """Calculate signal energy using PyTorch."""
    signal_torch = torch.tensor(signal, dtype=torch.float32)
    return torch.sum(signal_torch ** 2).item()

def numpy_above_mean_density(signal):
    """Calculate above mean density using NumPy."""
    mean_val = np.mean(signal)
    above_count = np.sum(signal > mean_val)
    return above_count / len(signal)

def tensorflow_above_mean_density(signal):
    """Calculate above mean density using TensorFlow."""
    signal_tf = tf.convert_to_tensor(signal, dtype=tf.float32)
    mean_val = tf.reduce_mean(signal_tf)
    above_count = tf.reduce_sum(tf.cast(signal_tf > mean_val, tf.float32))
    return (above_count / len(signal)).numpy()

def pytorch_above_mean_density(signal):
    """Calculate above mean density using PyTorch."""
    signal_torch = torch.tensor(signal, dtype=torch.float32)
    mean_val = torch.mean(signal_torch)
    above_count = torch.sum(signal_torch > mean_val)
    return (above_count / len(signal)).item()

def numpy_max_locations(signal):
    """Find first and last max locations using NumPy."""
    max_val = np.max(signal)
    max_indices = np.where(signal == max_val)[0]
    first_max = max_indices[0] if len(max_indices) > 0 else -1
    last_max = max_indices[-1] if len(max_indices) > 0 else -1
    return first_max, last_max

def tensorflow_max_locations(signal):
    """Find first and last max locations using TensorFlow."""
    signal_tf = tf.convert_to_tensor(signal, dtype=tf.float32)
    max_val = tf.reduce_max(signal_tf)
    max_mask = tf.equal(signal_tf, max_val)
    max_indices = tf.where(max_mask)[:, 0]
    first_max = max_indices[0].numpy() if len(max_indices) > 0 else -1
    last_max = max_indices[-1].numpy() if len(max_indices) > 0 else -1
    return int(first_max), int(last_max)

def pytorch_max_locations(signal):
    """Find first and last max locations using PyTorch."""
    signal_torch = torch.tensor(signal, dtype=torch.float32)
    max_val = torch.max(signal_torch)
    max_mask = signal_torch == max_val
    max_indices = torch.where(max_mask)[0]
    first_max = max_indices[0].item() if len(max_indices) > 0 else -1
    last_max = max_indices[-1].item() if len(max_indices) > 0 else -1
    return int(first_max), int(last_max)

def numpy_mean_change_properties(signal):
    """Calculate mean change properties using NumPy."""
    if len(signal) < 2:
        return 0.0, 0.0, 0.0
    
    changes = np.diff(signal)
    mean_change = np.mean(changes)
    mean_abs_change = np.mean(np.abs(changes))
    mean_squared_change = np.mean(changes ** 2)
    
    return mean_change, mean_abs_change, mean_squared_change

def tensorflow_mean_change_properties(signal):
    """Calculate mean change properties using TensorFlow."""
    if len(signal) < 2:
        return 0.0, 0.0, 0.0
    
    signal_tf = tf.convert_to_tensor(signal, dtype=tf.float32)
    changes = signal_tf[1:] - signal_tf[:-1]
    mean_change = tf.reduce_mean(changes).numpy()
    mean_abs_change = tf.reduce_mean(tf.abs(changes)).numpy()
    mean_squared_change = tf.reduce_mean(changes ** 2).numpy()
    
    return mean_change, mean_abs_change, mean_squared_change

def pytorch_mean_change_properties(signal):
    """Calculate mean change properties using PyTorch."""
    if len(signal) < 2:
        return 0.0, 0.0, 0.0
    
    signal_torch = torch.tensor(signal, dtype=torch.float32)
    changes = signal_torch[1:] - signal_torch[:-1]
    mean_change = torch.mean(changes).item()
    mean_abs_change = torch.mean(torch.abs(changes)).item()
    mean_squared_change = torch.mean(changes ** 2).item()
    
    return mean_change, mean_abs_change, mean_squared_change

def calculate_all_properties_python(signal, method='numpy'):
    """Calculate all properties using specified Python method."""
    if method == 'numpy':
        energy = numpy_signal_energy(signal)
        density = numpy_above_mean_density(signal)
        first_max, last_max = numpy_max_locations(signal)
        mean_change, mean_abs_change, mean_squared_change = numpy_mean_change_properties(signal)
    elif method == 'tensorflow':
        energy = tensorflow_signal_energy(signal)
        density = tensorflow_above_mean_density(signal)
        first_max, last_max = tensorflow_max_locations(signal)
        mean_change, mean_abs_change, mean_squared_change = tensorflow_mean_change_properties(signal)
    elif method == 'pytorch':
        energy = pytorch_signal_energy(signal)
        density = pytorch_above_mean_density(signal)
        first_max, last_max = pytorch_max_locations(signal)
        mean_change, mean_abs_change, mean_squared_change = pytorch_mean_change_properties(signal)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'signal_energy': energy,
        'above_mean_density': density,
        'first_max_location': first_max,
        'last_max_location': last_max,
        'mean_change': mean_change,
        'mean_abs_change': mean_abs_change,
        'mean_squared_change': mean_squared_change
    }

def compare_implementations():
    """Compare C, NumPy, TensorFlow, and PyTorch implementations."""
    print("=== Signal Properties Implementation Comparison ===\n")
    
    # Generate test signals
    signals, t = generate_test_signals()
    
    # Test case names
    test_names = [
        "Exponential Decay with Noise",
        "Chirp Signal (Frequency Sweep)",
        "Step Function with Transitions",
        "Random Walk with Trend"
    ]
    
    results = {}
    
    for test_num in range(1, 5):
        test_name = test_names[test_num - 1]
        print(f"Testing: {test_name}")
        print("-" * 50)
        
        # Check if C output files exist
        c_properties_file = f"test_properties_{test_num}.txt"
        c_signal_file = f"test_signal_properties_{test_num}.txt"
        
        if not os.path.exists(c_properties_file) or not os.path.exists(c_signal_file):
            print(f"Warning: C output files not found for test {test_num}. Run the C implementation first.")
            continue
        
        # Load C results
        c_properties = load_c_properties(c_properties_file)
        c_signal = load_c_signal(c_signal_file)
        
        if c_properties is None or c_signal is None:
            continue
        
        # Get Python signal (should match C signal)
        python_signal = signals[test_num]
        
        # Calculate Python results
        numpy_props = calculate_all_properties_python(python_signal, 'numpy')
        tf_props = calculate_all_properties_python(python_signal, 'tensorflow')
        torch_props = calculate_all_properties_python(python_signal, 'pytorch')
        
        # Store results
        results[test_num] = {
            'name': test_name,
            'c': c_properties,
            'numpy': numpy_props,
            'tensorflow': tf_props,
            'pytorch': torch_props,
            'signal': python_signal
        }
        
        # Compare results
        print(f"Signal length: {len(python_signal)}")
        print(f"Signal range: [{np.min(python_signal):.3f}, {np.max(python_signal):.3f}]")
        print()
        
        # Property comparisons
        properties = ['signal_energy', 'above_mean_density', 'first_max_location', 
                     'last_max_location', 'mean_change', 'mean_abs_change', 'mean_squared_change']
        
        for prop in properties:
            c_val = c_properties[prop]
            np_val = numpy_props[prop]
            tf_val = tf_props[prop]
            torch_val = torch_props[prop]
            
            print(f"{prop}:")
            print(f"  C:         {c_val:.6e}" if isinstance(c_val, float) else f"  C:         {c_val}")
            print(f"  NumPy:     {np_val:.6e}" if isinstance(np_val, float) else f"  NumPy:     {np_val}")
            print(f"  TensorFlow: {tf_val:.6e}" if isinstance(tf_val, float) else f"  TensorFlow: {tf_val}")
            print(f"  PyTorch:   {torch_val:.6e}" if isinstance(torch_val, float) else f"  PyTorch:   {torch_val}")
            
            # Calculate relative differences for numeric properties
            if isinstance(c_val, float) and isinstance(np_val, float):
                rel_diff_np = abs(c_val - np_val) / (abs(c_val) + 1e-10)
                rel_diff_tf = abs(c_val - tf_val) / (abs(c_val) + 1e-10)
                rel_diff_torch = abs(c_val - torch_val) / (abs(c_val) + 1e-10)
                
                print(f"  Rel. diff (C vs NumPy):     {rel_diff_np:.6e}")
                print(f"  Rel. diff (C vs TensorFlow): {rel_diff_tf:.6e}")
                print(f"  Rel. diff (C vs PyTorch):   {rel_diff_torch:.6e}")
            print()
        
        print()
    
    # Create comparison plots
    create_comparison_plots(results, t)

def create_comparison_plots(results, t):
    """Create comparison plots for all test cases."""
    print("Creating comparison plots...")
    
    n_tests = len(results)
    if n_tests == 0:
        print("No results to plot.")
        return
    
    fig, axes = plt.subplots(n_tests, 2, figsize=(15, 4 * n_tests))
    if n_tests == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Signal Properties Implementation Comparison", fontsize=16)
    
    for i, (test_num, result) in enumerate(results.items()):
        test_name = result['name']
        signal = result['signal']
        
        # Plot time domain
        axes[i, 0].plot(t, signal, 'b-', linewidth=1)
        axes[i, 0].set_title(f"{test_name} - Time Domain")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)
        
        # Add markers for max locations
        c_props = result['c']
        first_max = int(c_props['first_max_location'])
        last_max = int(c_props['last_max_location'])
        if first_max >= 0 and first_max < len(t):
            axes[i, 0].axvline(x=t[first_max], color='r', linestyle='--', alpha=0.7, label=f'First max: {first_max}')
        if last_max >= 0 and last_max != first_max and last_max < len(t):
            axes[i, 0].axvline(x=t[last_max], color='g', linestyle='--', alpha=0.7, label=f'Last max: {last_max}')
        axes[i, 0].legend()
        
        # Plot property comparison
        properties = ['signal_energy', 'above_mean_density', 'mean_change', 'mean_abs_change']
        property_names = ['Energy', 'Above Mean Density', 'Mean Change', 'Mean Abs Change']
        
        c_values = [result['c'][prop] for prop in properties]
        np_values = [result['numpy'][prop] for prop in properties]
        tf_values = [result['tensorflow'][prop] for prop in properties]
        torch_values = [result['pytorch'][prop] for prop in properties]
        
        x = np.arange(len(properties))
        width = 0.2
        
        axes[i, 1].bar(x - 1.5*width, c_values, width, label='C', alpha=0.8)
        axes[i, 1].bar(x - 0.5*width, np_values, width, label='NumPy', alpha=0.8)
        axes[i, 1].bar(x + 0.5*width, tf_values, width, label='TensorFlow', alpha=0.8)
        axes[i, 1].bar(x + 1.5*width, torch_values, width, label='PyTorch', alpha=0.8)
        
        axes[i, 1].set_title(f"{test_name} - Property Comparison")
        axes[i, 1].set_xlabel("Properties")
        axes[i, 1].set_ylabel("Values")
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(property_names, rotation=45)
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('signal_properties_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'signal_properties_comparison.png'")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

def main():
    """Main function."""
    print("Signal Properties Validation Script")
    print("==================================\n")
    
    # Check if required files exist
    required_files = []
    for i in range(1, 5):
        required_files.extend([f"test_properties_{i}.txt", f"test_signal_properties_{i}.txt"])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing C implementation output files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run the C implementation first:")
        print("  make test-properties")
        print("\nOr manually:")
        print("  gcc -o signal_properties_test signal_properties.c -lm")
        print("  ./signal_properties_test")
        return 1
    
    # Run comparison
    compare_implementations()
    
    print("Validation completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
