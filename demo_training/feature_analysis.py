import tensorflow as tf
from dataset import SeismicDataset
from dataset_utils import create_mapping_vehicle_name_to_file_path, filter_samples_by_max_distance
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
from scipy import signal
from scipy.stats import f_oneway
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

FEATURE_DEFINITIONS = {
    'total_power': {
        'description': 'Sum of all PSD values',
        'type': 'power',
        'requires_psd': True
    },
    'above_mean_density': {
        'description': 'Proportion of PSD values above mean',
        'type': 'ratio',
        'requires_psd': True
    },
    'mean_change': {
        'description': 'Mean of PSD derivatives',
        'type': 'change',
        'requires_psd': True
    },
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
    'psd_25hz': {
        'description': 'PSD value at 25 Hz',
        'type': 'power',
        'requires_psd': True,
        'requires_freqs': True,
        'target_freq': 25
    },
    'psd_30hz': {
        'description': 'PSD value at 30 Hz',
        'type': 'power',
        'requires_psd': True,
        'requires_freqs': True,
        'target_freq': 30
    },
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
    'nperseg': 10,
    'noverlap': 5,
    'nfft': 20,
    'window': 'hann',
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
    
    # Compute PSD using Welch's method on full-scale data
    freqs, psd = signal.welch(
        sample_np,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=WELCH_PARAMS['window'],
        scaling=WELCH_PARAMS['scaling'],
        detrend=WELCH_PARAMS['detrend']
    )
    
    return freqs, psd


# ============================================================================
# INDIVIDUAL FEATURE CALCULATORS (One function per feature)
# ============================================================================

def calc_total_power(psd):
    """Calculate total power (sum of all PSD values)."""
    return np.sum(psd)


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
    
    # Ratio and change features (use PSD directly)
    features['total_power'] = calc_total_power(psd)
    features['above_mean_density'] = calc_above_mean_density(psd)
    features['mean_change'] = calc_mean_change(psd)
    features['mean_abs_change'] = calc_mean_abs_change(psd)
    features['spectral_centroid'] = calc_spectral_centroid(psd, freqs)
    
    # Frequency-specific PSD features
    for target_freq in [25, 30, 35, 40, 45]:
        feature_name = f'psd_{target_freq}hz'
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
        print(f"  Feature shape: {len(all_features['total_power'])}")
    
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
# FEATURE ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_feature_discriminant_power(features_dict, feature_names, verbose=True):
    """
    Analyze which features have the most discriminant power between vehicle classes.
    
    Args:
        features_dict: dict {feature_name: np.array}
        feature_names: list of feature names to analyze
        verbose: Print detailed results
    
    Returns:
        dict: {feature_name: {'f_statistic': f_val, 'p_value': p_val, 'discriminant_power': power}}
    """
    from scipy.stats import f_oneway
    
    vehicle_labels = features_dict['vehicle_labels']
    unique_vehicles = np.unique(vehicle_labels)
    
    if verbose:
        print(f"\nAnalyzing discriminant power of {len(feature_names)} features...")
        print(f"Vehicle classes: {unique_vehicles}")
        print("-" * 80)
    
    discriminant_results = {}
    
    for feature_name in feature_names:
        if feature_name == 'vehicle_labels':
            continue
            
        feature_values = features_dict[feature_name]
        
        # Group feature values by vehicle class
        vehicle_groups = [feature_values[vehicle_labels == vehicle] for vehicle in unique_vehicles]
        
        # Perform one-way ANOVA
        f_statistic, p_value = f_oneway(*vehicle_groups)
        
        # Calculate discriminant power (higher F-statistic = more discriminant)
        discriminant_power = f_statistic
        
        discriminant_results[feature_name] = {
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'discriminant_power': float(discriminant_power),
            'significant': p_value < 0.05
        }
        
        if verbose:
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature_name:20s}: F={f_statistic:8.2f}, p={p_value:.2e} {significance}")
    
    # Sort by discriminant power
    sorted_features = sorted(discriminant_results.items(), 
                           key=lambda x: x[1]['discriminant_power'], 
                           reverse=True)
    
    if verbose:
        print("-" * 80)
        print("TOP DISCRIMINANT FEATURES (sorted by F-statistic):")
        for i, (feature_name, results) in enumerate(sorted_features[:5]):
            print(f"{i+1:2d}. {feature_name:20s}: F={results['f_statistic']:8.2f}, p={results['p_value']:.2e}")
    
    return discriminant_results


def visualize_feature_distributions(features_dict, feature_names, top_n=6, save_path='feature_distributions.png'):
    """
    Create box plots showing feature distributions by vehicle class.
    
    Args:
        features_dict: dict {feature_name: np.array}
        feature_names: list of feature names to visualize
        top_n: number of top discriminant features to show
        save_path: where to save the plot
    """
    # Get discriminant power for feature selection
    discriminant_results = analyze_feature_discriminant_power(features_dict, feature_names, verbose=False)
    
    # Select top N most discriminant features
    sorted_features = sorted(discriminant_results.items(), 
                           key=lambda x: x[1]['discriminant_power'], 
                           reverse=True)
    top_features = [name for name, _ in sorted_features[:top_n]]
    
    # Create subplots
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    vehicle_labels = features_dict['vehicle_labels']
    unique_vehicles = np.unique(vehicle_labels)
    
    for i, feature_name in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Prepare data for box plot
        feature_values = features_dict[feature_name]
        data_for_plot = [feature_values[vehicle_labels == vehicle] for vehicle in unique_vehicles]
        
        # Create box plot
        bp = ax.boxplot(data_for_plot, labels=unique_vehicles, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vehicles)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{feature_name}\nF={discriminant_results[feature_name]["f_statistic"]:.2f}')
        ax.set_xlabel('Vehicle Class')
        ax.set_ylabel('Feature Value')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Feature distribution plots saved to: {save_path}")


def create_feature_correlation_heatmap(features_dict, feature_names, save_path='feature_correlation.png'):
    """
    Create correlation heatmap between features.
    
    Args:
        features_dict: dict {feature_name: np.array}
        feature_names: list of feature names
        save_path: where to save the plot
    """
    # Create DataFrame for correlation analysis
    data_for_corr = {}
    for feature_name in feature_names:
        if feature_name != 'vehicle_labels':
            data_for_corr[feature_name] = features_dict[feature_name]
    
    df = pd.DataFrame(data_for_corr)
    correlation_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Feature correlation heatmap saved to: {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_seismic_data_pipeline(train_data, val_data, 
                                  save_stats=True, stats_filepath='feature_statistics.json',
                                  verbose=True):
    """
    Complete pipeline: Extract features from RAW data, normalize, prepare for NN.
    
    Args:
        train_data: dict {vehicle_type: [samples]}
        val_data: dict {vehicle_type: [samples]}
        save_stats: If True, save statistics to file
        stats_filepath: Where to save/load feature statistics
        verbose: Print progress
    
    Returns:
        dict containing:
            'X_train': np.array [n_train, n_features]
            'X_val': np.array [n_val, n_features]
            'y_train': np.array vehicle labels
            'y_val': np.array vehicle labels
            'feature_names': list of feature names
            'feature_statistics': dict of training statistics
    """
    print("\n" + "="*70)
    print("SEISMIC FEATURE EXTRACTION & NORMALIZATION PIPELINE")
    print("="*70)
    
    # Step 1: Extract features from RAW training data
    if verbose:
        print("\n[STEP 1] Extracting features from RAW data...")
    train_features_raw = extract_all_features_from_data(train_data, verbose=verbose)
    
    # Step 2: Extract features from RAW validation data
    if verbose:
        print("\n[STEP 2] Extracting features from RAW validation data...")
    val_features_raw = extract_all_features_from_data(val_data, verbose=verbose)
    
    # Step 3: Compute normalization statistics from TRAINING data ONLY
    if verbose:
        print("\n[STEP 3] Computing normalization parameters from TRAINING data...")
    train_statistics = compute_feature_statistics(train_features_raw, verbose=verbose)
    
    # Step 4: Normalize both training and validation using TRAINING statistics
    if verbose:
        print("\n[STEP 4] Normalizing features to [0, 1]...")
    train_features_normalized = normalize_features(
        train_features_raw, 
        train_statistics, 
        verbose=verbose
    )
    val_features_normalized = normalize_features(
        val_features_raw, 
        train_statistics,  # Use TRAINING statistics (no data leakage!)
        verbose=verbose
    )
    
    # Step 5: Convert to arrays
    if verbose:
        print("\n[STEP 5] Converting to neural network format...")
    X_train, feature_names = features_dict_to_array(train_features_normalized)
    X_val, _ = features_dict_to_array(val_features_normalized, feature_order=feature_names)
    
    # Extract labels
    y_train = train_features_normalized['vehicle_labels']
    y_val = val_features_normalized['vehicle_labels']
    
    if verbose:
        print(f"✓ X_train shape: {X_train.shape}")
        print(f"✓ X_val shape: {X_val.shape}")
        print(f"✓ Features: {feature_names}")
    
    # Step 6: Save statistics for future use
    if save_stats:
        if verbose:
            print(f"\n[STEP 6] Saving feature statistics...")
        save_feature_statistics(train_statistics, stats_filepath)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'feature_names': feature_names,
        'feature_statistics': train_statistics
    }


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



    

if __name__ == "__main__":
    config_path = 'feature_analysis_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_mapping, val_mapping = get_mapped_dataset(config)
    included_classes = config['vehicle_classification']['included_classes']
    print("After filtering by max distance train mapping:")
    print_count_from_mapping(train_mapping)
    print("After filtering by max distance val mapping:")
    print_count_from_mapping(val_mapping)

    print("Loading data...")
    train_data = load_data(train_mapping)
    val_data = load_data(val_mapping)
    print("Data loaded successfully!")
    print("Printing data shape...")
    print("Train data shape:")
    print_data_shape(train_data)
    print("Val data shape:")
    print_data_shape(val_data)

    print("Reshaping data to 2D...")
    train_data = reshape_data_to_2d(train_data)
    val_data = reshape_data_to_2d(val_data)
    print("Data reshaped successfully!")
    print("Printing data shape...")
    print("Train data shape:")
    print_data_shape(train_data)
    print("Val data shape:")
    print_data_shape(val_data)
    
    # ============================================================================
    # NEW FEATURE ANALYSIS PIPELINE
    # ============================================================================
    
    print("\n" + "="*70)
    print("SEISMIC FEATURE EXTRACTION & ANALYSIS PIPELINE")
    print("="*70)
    
    # Run the complete feature extraction pipeline
    results = process_seismic_data_pipeline(
        train_data, 
        val_data,
        save_stats=True,
        stats_filepath='feature_statistics.json',
        verbose=True
    )
    
    # Extract results
    X_train = results['X_train']
    X_val = results['X_val']
    y_train = results['y_train']
    y_val = results['y_val']
    feature_names = results['feature_names']
    feature_statistics = results['feature_statistics']
    
    print(f"\n✓ Pipeline Results:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Feature names: {feature_names}")
    
    # ============================================================================
    # FEATURE DISCRIMINANT ANALYSIS
    # ============================================================================
    
    print("\n" + "="*70)
    print("FEATURE DISCRIMINANT ANALYSIS")
    print("="*70)
    
    # Create features dict for analysis (using normalized training data)
    train_features_for_analysis = {}
    for i, feature_name in enumerate(feature_names):
        train_features_for_analysis[feature_name] = X_train[:, i]
    train_features_for_analysis['vehicle_labels'] = y_train
    
    # Analyze discriminant power
    discriminant_results = analyze_feature_discriminant_power(
        train_features_for_analysis, 
        feature_names, 
        verbose=True
    )
    
    # ============================================================================
    # FEATURE VISUALIZATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("FEATURE VISUALIZATION")
    print("="*70)
    
    # Create feature distribution plots
    print("Creating feature distribution plots...")
    visualize_feature_distributions(
        train_features_for_analysis, 
        feature_names, 
        top_n=6, 
        save_path='feature_distributions.png'
    )
    
    # Create correlation heatmap
    print("Creating feature correlation heatmap...")
    create_feature_correlation_heatmap(
        train_features_for_analysis, 
        feature_names, 
        save_path='feature_correlation.png'
    )
    
    # ============================================================================
    # SUMMARY REPORT
    # ============================================================================
    
    print("\n" + "="*70)
    print("FEATURE ANALYSIS SUMMARY")
    print("="*70)
    
    # Get top discriminant features
    sorted_features = sorted(discriminant_results.items(), 
                           key=lambda x: x[1]['discriminant_power'], 
                           reverse=True)
    
    print(f"\nTop 5 Most Discriminant Features:")
    for i, (feature_name, results) in enumerate(sorted_features[:5]):
        significance = "***" if results['p_value'] < 0.001 else "**" if results['p_value'] < 0.01 else "*" if results['p_value'] < 0.05 else ""
        print(f"{i+1:2d}. {feature_name:20s}: F={results['f_statistic']:8.2f}, p={results['p_value']:.2e} {significance}")
    
    # Count significant features
    significant_features = [name for name, results in discriminant_results.items() 
                          if results['significant']]
    print(f"\nSignificant features (p < 0.05): {len(significant_features)}/{len(feature_names)}")
    print(f"Significant features: {significant_features}")
    
    print(f"\n✓ Feature analysis completed successfully!")
    print(f"✓ Results saved:")
    print(f"  - Feature statistics: feature_statistics.json")
    print(f"  - Distribution plots: feature_distributions.png")
    print(f"  - Correlation heatmap: feature_correlation.png")
    print(f"✓ Data ready for neural network training:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_val shape: {X_val.shape}")
    print(f"  - All features normalized to [0, 1]")
    print(f"  - No data leakage (validation normalized using training statistics)")