#!/usr/bin/env python3
"""
Generate test data for Arduino queue testing from real training dataset.
Extracts 5 nissan samples with 200 data points each and saves as C header file.
"""

import numpy as np
import yaml
import os
import sys
import torch

# Add current directory to path to import feature_utils
sys.path.append(os.path.dirname(__file__))
from feature_utils import get_mapped_dataset, load_sample

def load_nissan_samples(num_samples=5):
    """Load multiple nissan samples from the ML dataset."""
    # Load configuration
    config_path = 'train_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset mappings
    train_mapping, _ = get_mapped_dataset(config)
    
    # Get nissan file paths
    if 'nissan' not in train_mapping or len(train_mapping['nissan']) == 0:
        raise ValueError("No nissan samples found in training data")
    
    nissan_files = train_mapping['nissan']
    if len(nissan_files) < num_samples:
        print(f"Warning: Only {len(nissan_files)} nissan files available, using all of them")
        num_samples = len(nissan_files)
    
    samples = []
    for i in range(num_samples):
        file_path = nissan_files[i]
        print(f"Loading nissan sample {i+1}/{num_samples} from: {file_path}")
        
        # Load the sample
        sample = load_sample(file_path)
        
        # Convert to numpy array and flatten if needed
        if isinstance(sample, torch.Tensor):
            sample_np = sample.numpy()
        else:
            sample_np = np.asarray(sample)
        
        # Flatten to 1D
        if sample_np.ndim > 1:
            sample_np = sample_np.flatten()
        
        # Take first 200 points
        if len(sample_np) >= 200:
            sample_200 = sample_np[:200]
        else:
            # Pad with zeros if sample is too short
            sample_200 = np.pad(sample_np, (0, 200 - len(sample_np)), 'constant')
        
        samples.append(sample_200)
        print(f"  Sample {i+1} shape: {sample_200.shape}")
        print(f"  Sample {i+1} range: [{np.min(sample_200):.6f}, {np.max(sample_200):.6f}]")
    
    return samples

def generate_c_header(samples, output_path):
    """Generate C header file with test data arrays."""
    with open(output_path, 'w') as f:
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")
        f.write("// Test data arrays for queue testing\n")
        f.write("// Generated from real nissan vehicle samples\n\n")
        
        for i, sample in enumerate(samples):
            f.write(f"// Test sample {i+1} - {len(sample)} data points\n")
            f.write(f"static const float test_data_{i+1}[200] = {{\n")
            
            # Format as C array with proper indentation
            for j in range(0, len(sample), 8):  # 8 values per line
                line_values = []
                for k in range(8):
                    if j + k < len(sample):
                        line_values.append(f"{sample[j + k]:.10f}f")
                    else:
                        break
                
                if j + 8 < len(sample):
                    f.write(f"    {', '.join(line_values)},\n")
                else:
                    f.write(f"    {', '.join(line_values)}\n")
            
            f.write("};\n\n")
        
        f.write(f"// Number of test samples available\n")
        f.write(f"#define NUM_TEST_SAMPLES {len(samples)}\n\n")
        f.write("#endif // TEST_DATA_H\n")

def main():
    """Main function to generate test data."""
    print("Generating test data for Arduino queue testing...")
    print("=" * 50)
    
    try:
        # Load 5 nissan samples
        samples = load_nissan_samples(50)
        
        # Generate C header file
        output_path = '../arduino_code_r1/test_large_data.h'
        generate_c_header(samples, output_path)
        
        print(f"\nTest data generated successfully!")
        print(f"Output file: {output_path}")
        print(f"Generated {len(samples)} test arrays with 200 data points each")
        
    except Exception as e:
        print(f"Error generating test data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
