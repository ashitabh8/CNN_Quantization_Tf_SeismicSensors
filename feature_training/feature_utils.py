import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import Counter
import warnings
import torch
from scipy.stats import f_oneway
from scipy.signal import welch
import json
import warnings
warnings.filterwarnings('ignore')

# Add demo_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demo_training'))
from dataset_utils import create_mapping_vehicle_name_to_file_path, filter_samples_by_max_distance


# ============================================================================
# SCIPY WELCH IMPLEMENTATION
# ============================================================================

def scipy_welch(signal, sampling_rate=100, nperseg=50, noverlap=25, nfft=64, 
                scaling='density', detrend='constant', window='hann'):
    """
    Apply Welch's method using scipy.signal.welch.
    
    Args:
        signal: Input signal array
        sampling_rate: Sampling frequency in Hz
        nperseg: Length of each segment
        noverlap: Number of overlapping points
        nfft: FFT length
        scaling: 'density' (V**2/Hz) or 'spectrum' (V**2)
        detrend: 'constant', 'linear', or False
        window: Window type ('hann', 'boxcar', etc.)
    
    Returns:
        np.array: Power spectral density or power spectrum values
    """
    # Convert to numpy array
    signal_np = np.asarray(signal, dtype=np.float32)
    
    # Use scipy's welch function
    freqs, psd = welch(
        signal_np,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        detrend=detrend,
        window=window,
        return_onesided=True
    )
    
    return psd



# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

FEATURE_DEFINITIONS = {
    'total_power': {
        'description': 'Total power in frequency domain (sum of PSD values)',
        'type': 'power',
        'requires_psd': True
    },
    # 'above_mean_density': {
    #     'description': 'Proportion of PSD values above mean',
    #     'type': 'ratio',
    #     'requires_psd': True
    # },
    # 'mean_change': {
    #     'description': 'Mean of PSD derivatives',
    #     'type': 'change',
    #     'requires_psd': True
    # },
    'mean_abs_change': {
        'description': 'Mean of absolute PSD derivatives',
        'type': 'change',
        'requires_psd': True
    },
    'spectral_centroid': {
        'description': 'Weighted average frequency',
        'type': 'frequency',
        'requires_psd': True,
        'requires_freqs': True
    },
    # 'psd_25hz': {
    #     'description': 'PSD value at 25 Hz',
    #     'type': 'power',
    #     'requires_psd': True,
    #     'requires_freqs': True,
    #     'target_freq': 25
    # },
    # 'psd_30hz': {
    #     'description': 'PSD value at 30 Hz',
    #     'type': 'power',
    #     'requires_psd': True,
    #     'requires_freqs': True,
    #     'target_freq': 30
    # },
    'psd_35hz': {
        'description': 'PSD value at 35 Hz',
        'type': 'power',
        'requires_psd': True,
        'requires_freqs': True,
        'target_freq': 35
    },
    'psd_40hz': {
        'description': 'PSD value at 40 Hz',
        'type': 'power',
        'requires_psd': True,
        'requires_freqs': True,
        'target_freq': 40
    },
    'psd_45hz': {
        'description': 'PSD value at 45 Hz',
        'type': 'power',
        'requires_psd': True,
        'requires_freqs': True,
        'target_freq': 45
    },
}

WELCH_PARAMS = {
    'sampling_rate': 100,
    'nperseg': 50,
    'noverlap': 25,
    'nfft': 64,
    'window': 'hann',  # Rectangular window (no windowing)
    'scaling': 'density',
    'detrend': 'constant'
}


# ============================================================================
# PSD COMPUTATION (Foundation for all features)
# ============================================================================

def compute_psd(sample, sampling_rate=WELCH_PARAMS['sampling_rate'],
                nperseg=WELCH_PARAMS['nperseg'],
                noverlap=WELCH_PARAMS['noverlap'],
                nfft=WELCH_PARAMS['nfft']):
    """
    Compute Power Spectral Density using Welch's method on RAW data.
    
    Args:
        sample: Input signal, shape [n_channels, n_samples] or [n_samples]
        sampling_rate: Sampling frequency in Hz
        nperseg: Length of each segment
        noverlap: Number of overlapping points
        nfft: FFT length
    
    Returns:
        freqs: Frequency bins (Hz)
        psd: Power spectral density values
    
    Raises:
        ValueError: If sample cannot be flattened to 1D
    """
    # Convert torch tensor to numpy if needed
    if isinstance(sample, torch.Tensor):
        sample_np = sample.numpy()
    else:
        sample_np = np.asarray(sample)
    
    # Flatten to 1D
    if sample_np.ndim > 1:
        sample_np = sample_np.flatten()
    
    if sample_np.ndim != 1:
        raise ValueError(f"Sample must be 1D after flattening, got shape: {sample_np.shape}")
    
    # Compute PSD using scipy's Welch's method
    psd = scipy_welch(
        sample_np,
        sampling_rate=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=WELCH_PARAMS['scaling'],
        detrend=WELCH_PARAMS['detrend'],
        window=WELCH_PARAMS['window']
    )
    
    # Generate frequency bins to match the expected output format
    freqs = np.linspace(0, sampling_rate/2, len(psd))
    
    return freqs, psd


# ============================================================================
# INDIVIDUAL FEATURE CALCULATORS (One function per feature)
# ============================================================================

def calc_total_power(psd, freqs=None):
    """
    Calculate total power in frequency domain using Power Spectral Density (PSD).
    
    Args:
        psd: Power spectral density values
        freqs: Frequency bins (optional, for validation)
    
    Returns:
        Total power as sum of PSD values
    """
    # Convert torch tensor to numpy if needed
    if isinstance(psd, torch.Tensor):
        psd_np = psd.numpy()
    else:
        psd_np = np.asarray(psd)
    
    # Flatten to 1D
    if psd_np.ndim > 1:
        psd_np = psd_np.flatten()
    
    # Calculate total power: sum of PSD values
    # This represents the total power across all frequency components
    return np.sum(psd_np)


def calc_above_mean_density(psd):
    """Calculate proportion of PSD values above the mean."""
    mean_psd = np.mean(psd)
    above_mean_count = np.sum(psd > mean_psd)
    density = above_mean_count / len(psd)
    return density


def calc_mean_change(psd):
    """Calculate mean of PSD first-order differences."""
    return np.mean(np.diff(psd))


def calc_mean_abs_change(psd):
    """Calculate mean of absolute PSD first-order differences."""
    return np.mean(np.abs(np.diff(psd)))


def calc_spectral_centroid(psd, freqs):
    """
    Calculate spectral centroid (weighted average frequency).
    
    Args:
        psd: Power spectral density values
        freqs: Frequency bins
    
    Returns:
        Centroid frequency in Hz
    """
    centroid = np.sum(freqs * psd) / np.sum(psd)
    return centroid


def calc_psd_at_frequency(psd, freqs, target_freq):
    """
    Get PSD value at or nearest to target frequency.
    
    Args:
        psd: Power spectral density values
        freqs: Frequency bins
        target_freq: Desired frequency in Hz
    
    Returns:
        PSD value at target frequency
    """
    idx = np.argmin(np.abs(freqs - target_freq))
    return psd[idx]


# ============================================================================
# FEATURE EXTRACTION FROM SINGLE SAMPLE
# ============================================================================

def extract_features_from_sample(sample, freqs, psd):
    """
    Extract all features from a single sample.
    
    Args:
        sample: Raw signal (for reference/debugging)
        freqs: Frequency bins from PSD computation
        psd: Power spectral density values
    
    Returns:
        dict: {feature_name: feature_value}
    """
    features = {}
    
    # Extract only features that are defined in FEATURE_DEFINITIONS
    for feature_name, feature_config in FEATURE_DEFINITIONS.items():
        if feature_name == 'total_power':
            features[feature_name] = calc_total_power(psd, freqs)
        elif feature_name == 'above_mean_density':
            features[feature_name] = calc_above_mean_density(psd)
        elif feature_name == 'mean_change':
            features[feature_name] = calc_mean_change(psd)
        elif feature_name == 'mean_abs_change':
            features[feature_name] = calc_mean_abs_change(psd)
        elif feature_name == 'spectral_centroid':
            features[feature_name] = calc_spectral_centroid(psd, freqs)
        elif feature_name.startswith('psd_') and feature_name.endswith('hz'):
            # Extract target frequency from feature name (e.g., 'psd_25hz' -> 25)
            target_freq = int(feature_name.split('_')[1].replace('hz', ''))
            features[feature_name] = calc_psd_at_frequency(psd, freqs, target_freq)
    
    return features


# ============================================================================
# BATCH FEATURE EXTRACTION (All data at once)
# ============================================================================

def extract_all_features_from_data(data_dict, verbose=True):
    """
    Extract all features from data dictionary.
    
    Args:
        data_dict: dict {vehicle_type: [samples]}
            Each sample should be shape [1, 200] or [200]
        verbose: Print progress information
    
    Returns:
        dict: {feature_name: np.array of shape [total_samples]}
              Also includes 'vehicle_labels' for tracking
    
    Example:
        >>> train_features = extract_all_features_from_data(train_data)
        >>> print(train_features.keys())
        ['total_power', 'above_mean_density', ..., 'vehicle_labels']
        >>> print(train_features['total_power'].shape)
        (1250,)
    """
    # Initialize containers for all features
    all_features = {feature_name: [] for feature_name in FEATURE_DEFINITIONS.keys()}
    all_features['vehicle_labels'] = []
    
    total_samples = sum(len(samples) for samples in data_dict.values())
    processed_count = 0
    freqs_cached = None
    
    if verbose:
        print(f"\nExtracting features from {total_samples} samples...")
        print(f"Features to extract: {list(FEATURE_DEFINITIONS.keys())}")
    
    # Process each vehicle type
    for vehicle_type, samples in data_dict.items():
        if verbose:
            print(f"  Processing {vehicle_type}: {len(samples)} samples...", end=" ")
        
        vehicle_feature_count = 0
        
        # Process each sample
        for sample in samples:
            try:
                # Compute PSD on RAW data (NO normalization yet!)
                freqs, psd = compute_psd(sample)
                
                # Cache frequency bins (should be identical for all samples)
                if freqs_cached is None:
                    freqs_cached = freqs
                
                # Extract all features from this sample
                sample_features = extract_features_from_sample(sample, freqs, psd)
                
                # Store features
                for feature_name, feature_value in sample_features.items():
                    all_features[feature_name].append(feature_value)
                
                all_features['vehicle_labels'].append(vehicle_type)
                vehicle_feature_count += 1
                processed_count += 1
                
            except Exception as e:
                print(f"\n    ERROR processing sample: {e}")
                continue
        
        if verbose:
            print(f"✓ {vehicle_feature_count} successful")
    
    # Convert lists to numpy arrays for easier manipulation
    for feature_name in FEATURE_DEFINITIONS.keys():
        all_features[feature_name] = np.array(all_features[feature_name])
    
    all_features['vehicle_labels'] = np.array(all_features['vehicle_labels'])
    
    if verbose:
        print(f"✓ Total samples processed: {processed_count}/{total_samples}")
        print(f"  Frequency range: {freqs_cached[0]:.2f} - {freqs_cached[-1]:.2f} Hz")
        # print(f"  Feature shape: {len(all_features['total_power'])}")
    
    return all_features


# ============================================================================
# FEATURE STATISTICS COMPUTATION
# ============================================================================

def compute_feature_statistics(features_dict, exclude_labels=True, verbose=True):
    """
    Compute min, max, mean, std for each feature from training data.
    IMPORTANT: Call this on TRAINING data only to avoid data leakage.
    
    Args:
        features_dict: Output from extract_all_features_from_data()
        exclude_labels: If True, skip 'vehicle_labels' key
        verbose: Print statistics
    
    Returns:
        dict: {feature_name: {'min': x, 'max': y, 'mean': m, 'std': s, 'range': r}}
    """
    statistics = {}
    
    if verbose:
        print("\nComputing feature statistics from TRAINING data...")
        print("-" * 70)
    
    for feature_name, values in features_dict.items():
        if exclude_labels and feature_name == 'vehicle_labels':
            continue
        
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        range_val = max_val - min_val
        
        statistics[feature_name] = {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val),
            'range': float(range_val),
            'count': len(values)
        }
        
        if verbose:
            print(f"{feature_name:20s}: [{min_val:.3e}, {max_val:.3e}] "
                  f"mean={mean_val:.3e}, std={std_val:.3e}")
    
    if verbose:
        print("-" * 70)
    
    return statistics


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

def normalize_single_feature(feature_values, feature_stats, feature_name):
    """
    Normalize a single feature to [0, 1] using pre-computed statistics.
    
    Args:
        feature_values: np.array of shape [n_samples]
        feature_stats: dict with 'min', 'max', 'range'
        feature_name: name of feature (for debugging)
    
    Returns:
        np.array: Normalized values in [0, 1]
    
    Raises:
        ValueError: If range is zero (feature is constant)
    """
    min_val = feature_stats['min']
    max_val = feature_stats['max']
    range_val = feature_stats['range']
    
    if range_val == 0:
        raise ValueError(
            f"Feature '{feature_name}' has zero range (min={min_val}, max={max_val}). "
            f"Cannot normalize. Feature is constant across all training samples."
        )
    
    # Min-max normalization: (x - min) / (max - min)
    normalized = (feature_values - min_val) / range_val
    
    # Clip to [0, 1] in case of small numerical errors or val data outside train range
    normalized = np.clip(normalized, 0, 1)
    
    return normalized


def normalize_features(features_dict, feature_statistics, verbose=True):
    """
    Normalize all features to [0, 1] using training statistics.
    IMPORTANT: Use feature_statistics computed from TRAINING data only.
    
    Args:
        features_dict: dict {feature_name: np.array}
        feature_statistics: dict from compute_feature_statistics()
        verbose: Print progress
    
    Returns:
        dict: {feature_name: normalized_np.array}
    
    Example:
        >>> train_features = extract_all_features_from_data(train_data)
        >>> train_stats = compute_feature_statistics(train_features)
        >>> train_normalized = normalize_features(train_features, train_stats)
        >>> val_normalized = normalize_features(val_features, train_stats)  # Use TRAIN stats!
    """
    normalized_features = {}
    
    if verbose:
        print("\nNormalizing features to [0, 1]...")
        print("-" * 70)
    
    for feature_name, feature_values in features_dict.items():
        if feature_name == 'vehicle_labels':
            # Don't normalize labels
            normalized_features[feature_name] = feature_values
            continue
        
        try:
            normalized = normalize_single_feature(
                feature_values,
                feature_statistics[feature_name],
                feature_name
            )
            normalized_features[feature_name] = normalized
            
            if verbose:
                print(f"{feature_name:20s}: [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")
        
        except KeyError:
            print(f"WARNING: Feature '{feature_name}' not found in statistics. Skipping.")
        except ValueError as e:
            print(f"ERROR normalizing '{feature_name}': {e}")
    
    if verbose:
        print("-" * 70)
    
    return normalized_features


# ============================================================================
# FEATURE ARRAY CONVERSION FOR NEURAL NETWORK
# ============================================================================

def filter_features_by_config(features_dict, features_to_use):
    """
    Filter features dictionary to only include specified features.
    
    Args:
        features_dict: dict {feature_name: np.array}
        features_to_use: list of feature names to keep
    
    Returns:
        dict: filtered features dictionary
    
    Example:
        >>> filtered_features = filter_features_by_config(train_features, ['total_power', 'mean_abs_change'])
        >>> print(filtered_features.keys())
        ['total_power', 'mean_abs_change', 'vehicle_labels']
    """
    # Always keep vehicle_labels
    filtered_features = {'vehicle_labels': features_dict['vehicle_labels']}
    
    # Add only the specified features
    for feature_name in features_to_use:
        if feature_name in features_dict:
            filtered_features[feature_name] = features_dict[feature_name]
        else:
            print(f"⚠️  WARNING: Feature '{feature_name}' not found in extracted features!")
            print(f"Available features: {list(features_dict.keys())}")
    
    return filtered_features


def features_dict_to_array(features_dict, feature_order=None):
    """
    Convert features dictionary to numpy array suitable for neural network.
    
    Args:
        features_dict: dict {feature_name: np.array}
        feature_order: list of feature names in desired order
                       If None, uses alphabetical order
    
    Returns:
        tuple: (feature_array, feature_names)
            feature_array: shape [n_samples, n_features]
            feature_names: list of feature names (for reference)
    
    Example:
        >>> X_train, feature_names = features_dict_to_array(train_normalized)
        >>> print(X_train.shape)
        (1250, 10)
        >>> print(feature_names)
        ['total_power', 'above_mean_density', ...]
    """
    if feature_order is None:
        # Exclude labels, use alphabetical order
        feature_order = sorted([k for k in features_dict.keys() if k != 'vehicle_labels'])
    
    # Stack features as columns
    feature_arrays = [features_dict[fname] for fname in feature_order]
    feature_array = np.column_stack(feature_arrays)
    
    return feature_array, feature_order


# ============================================================================
# SAVE/LOAD FEATURE STATISTICS (For future use on new data)
# ============================================================================

def save_feature_statistics(feature_statistics, filepath):
    """
    Save feature statistics to JSON for later use on new data.
    
    Args:
        feature_statistics: dict from compute_feature_statistics()
        filepath: path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(feature_statistics, f, indent=2)
    print(f"✓ Feature statistics saved to: {filepath}")


def load_feature_statistics(filepath):
    """
    Load feature statistics from JSON.
    
    Args:
        filepath: path to JSON file
    
    Returns:
        dict: feature statistics
    """
    with open(filepath, 'r') as f:
        statistics = json.load(f)
    print(f"✓ Feature statistics loaded from: {filepath}")
    return statistics


# ============================================================================
# FEATURE DATA SAVE/LOAD FUNCTIONS
# ============================================================================

def save_features(features_dict, filepath):
    """
    Save extracted features to a pickle file for later loading.
    
    Args:
        features_dict: dict containing extracted features
        filepath: path to save the features
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"✓ Features saved to: {filepath}")


def load_features(filepath):
    """
    Load extracted features from a pickle file.
    
    Args:
        filepath: path to the saved features file
    
    Returns:
        dict: loaded features
    """
    import pickle
    with open(filepath, 'rb') as f:
        features = pickle.load(f)
    print(f"✓ Features loaded from: {filepath}")
    return features


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def get_mapped_dataset(config):
    train_mapping, val_mapping = create_mapping_vehicle_name_to_file_path(config)
    train_mapping = filter_samples_by_max_distance(train_mapping, config['max_distance_m'])
    val_mapping = filter_samples_by_max_distance(val_mapping, config['max_distance_m'])
    return train_mapping, val_mapping


def print_count_from_mapping(mapping):
    for vehicle, file_paths in mapping.items():
        print(f"{vehicle}: {len(file_paths)}")


def load_sample(file_path):
    sample = torch.load(file_path, weights_only=False)
    return sample['data']['shake']['seismic']


def load_data(car_to_file_path_mapping):
    data = {}
    labels = car_to_file_path_mapping.keys()
    for label in labels:
        data[label] = []
    for vehicle, file_paths in car_to_file_path_mapping.items():
        for file_path in file_paths:
            data[vehicle].append(load_sample(file_path))
    return data


def print_data_shape(data):
    for vehicle, data in data.items():
        print(f"{vehicle} : data[0].shape = {data[0].shape}")


def reshape_data_to_2d(data):
    # reshape data from [1,10,20] to [1,200]
    reshaped_data = {}
    for vehicle in data.keys():
        reshaped_data[vehicle] = []
    for vehicle, data in data.items():
        for i, sample in enumerate(data):
            new_sample = sample.reshape(-1, sample.shape[1] * sample.shape[2])
            reshaped_data[vehicle].append(new_sample)
    return reshaped_data


def remove_background_from_training_data(data_dict, verbose=True):
    """
    Remove background samples from training data to avoid class imbalance.
    
    Args:
        data_dict: dict {vehicle_type: [samples]}
        verbose: Print removal statistics
    
    Returns:
        dict: Training data without background samples
    """
    filtered_data = {}
    
    if verbose:
        print("\nRemoving background samples from training data...")
        print("-" * 50)
    
    total_samples_before = sum(len(samples) for samples in data_dict.values())
    background_samples = len(data_dict.get('background', []))
    
    for vehicle_type, samples in data_dict.items():
        if vehicle_type != 'background':
            filtered_data[vehicle_type] = samples
            if verbose:
                print(f"  {vehicle_type:12s}: {len(samples):4d} samples (kept)")
        else:
            if verbose:
                print(f"  {vehicle_type:12s}: {len(samples):4d} samples (removed)")
    
    total_samples_after = sum(len(samples) for samples in filtered_data.values())
    
    if verbose:
        print(f"\nTraining data filtering summary:")
        print(f"  Samples before: {total_samples_before}")
        print(f"  Background removed: {background_samples}")
        print(f"  Samples after: {total_samples_after}")
        print(f"  Reduction: {((total_samples_before - total_samples_after) / total_samples_before * 100):.1f}%")
    
    return filtered_data


# ============================================================================
# CLASS BALANCING
# ============================================================================

def undersample_background(X_train, y_train, target_ratio=1.0, random_state=42, verbose=True):
    """
    Undersample the background class to balance the dataset.
    
    Args:
        X_train: Training features array (n_samples, n_features)
        y_train: Training labels array (n_samples, n_classes) - one-hot encoded
        target_ratio: Ratio of background samples to keep relative to other classes (default: 1.0)
        random_state: Random seed for reproducibility
        verbose: Whether to print balancing information
    
    Returns:
        tuple: (X_balanced, y_balanced) - balanced training data
    """
    import numpy as np
    from sklearn.utils import resample
    
    # Convert one-hot encoded labels back to class indices for analysis
    y_indices = np.argmax(y_train, axis=1)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y_indices, return_counts=True)
    
    if verbose:
        print(f"\nClass balancing analysis:")
        print(f"  Original class distribution:")
        for class_idx, count in zip(unique_classes, class_counts):
            print(f"    Class {class_idx}: {count} samples")
    
    # Find background class (assuming it's the first class with index 0)
    background_idx = 0  # Based on your config, background is the first class
    background_count = class_counts[unique_classes == background_idx][0]
    
    # Find the target number of background samples
    # Use the average of other classes as target
    other_classes = unique_classes[unique_classes != background_idx]
    other_counts = class_counts[unique_classes != background_idx]
    target_background_count = int(np.mean(other_counts) * target_ratio)
    
    if verbose:
        print(f"  Target background samples: {target_background_count}")
        print(f"  Background reduction: {background_count} -> {target_background_count}")
        print(f"  Reduction ratio: {target_background_count/background_count:.2f}")
    
    # Get indices of background samples
    background_indices = np.where(y_indices == background_idx)[0]
    
    # Randomly sample the target number of background samples
    np.random.seed(random_state)
    selected_background_indices = np.random.choice(
        background_indices, 
        size=target_background_count, 
        replace=False
    )
    
    # Get indices of all non-background samples
    non_background_indices = np.where(y_indices != background_idx)[0]
    
    # Combine selected background samples with all non-background samples
    balanced_indices = np.concatenate([selected_background_indices, non_background_indices])
    
    # Shuffle the balanced dataset
    np.random.seed(random_state)
    balanced_indices = np.random.permutation(balanced_indices)
    
    # Extract balanced data
    X_balanced = X_train[balanced_indices]
    y_balanced = y_train[balanced_indices]
    
    if verbose:
        # Verify the balancing
        y_balanced_indices = np.argmax(y_balanced, axis=1)
        balanced_unique, balanced_counts = np.unique(y_balanced_indices, return_counts=True)
        
        print(f"\n  Balanced class distribution:")
        for class_idx, count in zip(balanced_unique, balanced_counts):
            print(f"    Class {class_idx}: {count} samples")
        
        print(f"  Total samples: {len(X_balanced)} (reduced from {len(X_train)})")
        print(f"  Reduction: {((len(X_train) - len(X_balanced)) / len(X_train) * 100):.1f}%")
    
    return X_balanced, y_balanced


def balance_classes_by_undersampling(X_train, y_train, class_names, target_samples_per_class=None, 
                                   random_state=42, verbose=True):
    """
    Balance classes by undersampling the majority class(es).
    
    Args:
        X_train: Training features array (n_samples, n_features)
        y_train: Training labels array (n_samples, n_classes) - one-hot encoded
        class_names: List of class names
        target_samples_per_class: Target number of samples per class (if None, uses median of other classes)
        random_state: Random seed for reproducibility
        verbose: Whether to print balancing information
    
    Returns:
        tuple: (X_balanced, y_balanced) - balanced training data
    """
    import numpy as np
    
    # Convert one-hot encoded labels back to class indices
    y_indices = np.argmax(y_train, axis=1)
    
    # Get class counts
    unique_classes, class_counts = np.unique(y_indices, return_counts=True)
    
    if verbose:
        print(f"\nClass balancing by undersampling:")
        print(f"  Original distribution:")
        for i, (class_idx, count) in enumerate(zip(unique_classes, class_counts)):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            print(f"    {class_name}: {count} samples")
    
    # Determine target samples per class
    if target_samples_per_class is None:
        # Use the median of all classes as target
        target_samples_per_class = int(np.median(class_counts))
    
    if verbose:
        print(f"  Target samples per class: {target_samples_per_class}")
    
    # Collect balanced indices
    balanced_indices = []
    
    for class_idx in unique_classes:
        class_indices = np.where(y_indices == class_idx)[0]
        class_count = len(class_indices)
        
        if class_count > target_samples_per_class:
            # Undersample this class
            np.random.seed(random_state + class_idx)  # Different seed for each class
            selected_indices = np.random.choice(
                class_indices, 
                size=target_samples_per_class, 
                replace=False
            )
        else:
            # Keep all samples for this class
            selected_indices = class_indices
        
        balanced_indices.append(selected_indices)
        
        if verbose:
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            print(f"    {class_name}: {class_count} -> {len(selected_indices)} samples")
    
    # Combine all selected indices
    balanced_indices = np.concatenate(balanced_indices)
    
    # Shuffle the balanced dataset
    np.random.seed(random_state)
    balanced_indices = np.random.permutation(balanced_indices)
    
    # Extract balanced data
    X_balanced = X_train[balanced_indices]
    y_balanced = y_train[balanced_indices]
    
    if verbose:
        print(f"  Total samples: {len(X_balanced)} (reduced from {len(X_train)})")
        print(f"  Reduction: {((len(X_train) - len(X_balanced)) / len(X_train) * 100):.1f}%")
    
    return X_balanced, y_balanced
