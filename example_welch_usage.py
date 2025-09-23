#!/usr/bin/env python3
"""
Example script demonstrating how to use the new Welch's method for spectral processing
in the MultiModalDataset class.

This script shows how to:
1. Configure Welch's method parameters in the dataset config
2. Load the dataset with spectral processing enabled
3. Compare different spectral processing methods
"""

import yaml
import tensorflow as tf
import numpy as np

# Example configuration for using Welch's method
def create_welch_config():
    """Create a configuration that uses Welch's method for spectral processing."""
    config = {
        'vehicle_classification': {
            'num_classes': 10,
            'class_names': ["background", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            'included_classes': [0, 5, 8],
            'train_index_file': '/home/tkimura4/data/datasets/ACIDS/random_partition_index_vehicle_classification/train_index.txt',
            'val_index_file': '/home/tkimura4/data/datasets/ACIDS/random_partition_index_vehicle_classification/val_index.txt',
            'test_index_file': '/home/tkimura4/data/datasets/ACIDS/random_partition_index_vehicle_classification/test_index.txt'
        },
        
        # Spectral Processing Configuration
        'spectral_processing': {
            'method': 'welch',  # Use Welch's method instead of FFT
            'sampling_rate': 512.0,  # Sampling rate in Hz
            
            # Welch's Method Parameters
            # Data is pre-windowed with 7 segments of 256 samples each (half overlap)
            # We apply FFT to each segment and average across the 7 windows
            'welch': {
                'nperseg': 32,      # Not used - data is already windowed
                'noverlap': 16,     # Not used - data is already windowed with half overlap
                'nfft': 32,         # Length of the FFT used - gives 16 Hz bins (512 Hz / 32 = 16 Hz)
                'window': 'hann',   # Not used - data is already windowed
                'scaling': 'density',  # "density" for power spectral density, "spectrum" for power spectrum
                'detrend': False  # "constant", "linear", or False
            }
        }
    }
    return config

def create_fft_config():
    """Create a configuration that uses FFT for spectral processing (for comparison)."""
    config = create_welch_config()
    config['spectral_processing']['method'] = 'fft'
    return config

def create_no_spectral_config():
    """Create a configuration with no spectral processing."""
    config = create_welch_config()
    config['spectral_processing']['method'] = 'none'
    return config

class Args:
    """Simple args class to hold configuration."""
    def __init__(self, config):
        self.dataset_config = config
        self.task = "vehicle_classification"
        # Add spectral processing config to args
        self.spectral_processing = config.get('spectral_processing', {})

def demonstrate_spectral_processing():
    """Demonstrate different spectral processing methods."""
    
    print("=== Spectral Processing Demonstration ===\n")
    
    # Create different configurations
    welch_config = create_welch_config()
    fft_config = create_fft_config()
    no_spectral_config = create_no_spectral_config()
    
    print("1. Welch's Method Configuration:")
    print(f"   Method: {welch_config['spectral_processing']['method']}")
    print(f"   Window size (nperseg): {welch_config['spectral_processing']['welch']['nperseg']}")
    print(f"   Overlap: {welch_config['spectral_processing']['welch']['noverlap']}")
    print(f"   FFT length: {welch_config['spectral_processing']['welch']['nfft']}")
    print(f"   Window function: {welch_config['spectral_processing']['welch']['window']}")
    print(f"   Scaling: {welch_config['spectral_processing']['welch']['scaling']}")
    print(f"   Detrending: {welch_config['spectral_processing']['welch']['detrend']}")
    print()
    
    print("2. FFT Method Configuration:")
    print(f"   Method: {fft_config['spectral_processing']['method']}")
    print()
    
    print("3. No Spectral Processing Configuration:")
    print(f"   Method: {no_spectral_config['spectral_processing']['method']}")
    print()
    
    # Note: The actual dataset loading would require the index files to exist
    print("Note: To actually load the dataset, you would need to:")
    print("1. Ensure the index files exist at the specified paths")
    print("2. Import the MultiModalDataset class")
    print("3. Create dataset instances with the configurations above")
    print()
    
    print("Example usage:")
    print("""
    from dataset import MultiModalDataset
    
    # Create args with Welch configuration
    args = Args(welch_config)
    
    # Create dataset (assuming index files exist)
    train_dataset = MultiModalDataset(
        args=args,
        index_file=welch_config['vehicle_classification']['train_index_file']
    )
    
    # Convert to TensorFlow dataset
    train_tf_dataset = train_dataset.to_tf_dataset(batch_size=32)
    
    # The data will now be processed using Welch's method
    for data, label in train_tf_dataset.take(1):
        print(f"Processed data shape: {data.shape}")
        print(f"Label shape: {label.shape}")
        break
    """)

def explain_welch_parameters():
    """Explain the Welch's method parameters and their effects."""
    
    print("=== Welch's Method Parameters Explanation ===\n")
    
    print("RATIONALE FOR PRE-WINDOWED DATA PROCESSING:")
    print("The data is already windowed with 7 segments of 256 samples each (half overlap).")
    print("The Welch method implementation:")
    print("- Applies FFT to each of the 7 pre-windowed segments")
    print("- Averages the power spectral density across the 7 windows")
    print("- Uses nfft=32 to achieve 16 Hz bins (512 Hz / 32 = 16 Hz)")
    print("- Provides robustness through averaging across multiple windows")
    print()
    
    parameters = {
        'nperseg': {
            'description': 'Length of each segment (window size) - NOT USED',
            'effect': 'Data is already windowed with 7 segments of 256 samples each',
            'typical_range': 'N/A - data is pre-windowed',
            'recommendation': 'Not applicable - data is already windowed'
        },
        'noverlap': {
            'description': 'Number of points to overlap between segments - NOT USED',
            'effect': 'Data is already windowed with half overlap',
            'typical_range': 'N/A - data is pre-windowed',
            'recommendation': 'Not applicable - data is already windowed'
        },
        'nfft': {
            'description': 'Length of the FFT used for frequency analysis',
            'effect': 'Determines frequency bin size: bin_size = sampling_rate / nfft',
            'typical_range': '16-64 for 256-sample signals',
            'recommendation': '32 (gives 16 Hz bins: 512 Hz / 32 = 16 Hz)'
        },
        'window': {
            'description': 'Window function - NOT USED',
            'options': {
                'hann': 'Data is already windowed',
                'hamming': 'Data is already windowed',
                'blackman': 'Data is already windowed',
                'bartlett': 'Data is already windowed',
                'boxcar': 'Data is already windowed'
            },
            'recommendation': 'Not applicable - data is already windowed'
        },
        'scaling': {
            'description': 'Type of scaling applied to the result',
            'options': {
                'density': 'Power spectral density (PSD) - power per unit frequency',
                'spectrum': 'Power spectrum - total power in each frequency bin'
            },
            'recommendation': 'density (more standard for spectral analysis)'
        },
        'detrend': {
            'description': 'Type of detrending applied to each segment',
            'options': {
                'constant': 'Remove DC component (mean)',
                'linear': 'Remove linear trend',
                'False': 'No detrending (preserves all signal information)'
            },
            'recommendation': 'False (no detrending by default - preserves all signal information)'
        }
    }
    
    for param, info in parameters.items():
        print(f"{param.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Effect: {info['effect']}")
        if 'typical_range' in info:
            print(f"  Typical range: {info['typical_range']}")
        if 'recommendation' in info:
            print(f"  Recommendation: {info['recommendation']}")
        if 'options' in info:
            print(f"  Options:")
            for option, desc in info['options'].items():
                print(f"    {option}: {desc}")
        print()

if __name__ == "__main__":
    demonstrate_spectral_processing()
    print("\n" + "="*60 + "\n")
    explain_welch_parameters()
