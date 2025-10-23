#!/usr/bin/env python3
"""
Real-time seismic feature extraction system.

This script listens on UDP port 5005 for seismic sensor data from Arduino,
extracts features using the same pipeline as training, and normalizes them
using pre-computed statistics.
"""

import socket
import numpy as np
import json
import sys
import os
from datetime import datetime

# Import feature extraction utilities
from feature_utils import (
    compute_psd, extract_features_from_sample, normalize_features,
    features_dict_to_array, load_feature_statistics
)

# Configuration
UDP_PORT = 5005
BUFFER_SIZE = 2048
EXPECTED_SAMPLES = 200

# Paths to pre-computed statistics
FEATURE_STATS_PATH = '/home/misra8/CNN_Quantization_Tf_SeismicSensors/feature_training/experiments/FeatureModel_20251023_015914/feature_statistics.json'
BACKGROUND_THRESHOLD_PATH = '/home/misra8/CNN_Quantization_Tf_SeismicSensors/feature_training/experiments/FeatureModel_20251023_015914/background_threshold.json'

def load_statistics():
    """Load feature statistics and background threshold."""
    print("Loading feature statistics and background threshold...")
    
    # Load feature statistics
    feature_stats = load_feature_statistics(FEATURE_STATS_PATH)
    
    # Load background threshold
    with open(BACKGROUND_THRESHOLD_PATH, 'r') as f:
        background_data = json.load(f)
        background_threshold = background_data['threshold']
    
    print(f"✓ Feature statistics loaded: {len(feature_stats)} features")
    print(f"✓ Background threshold: {background_threshold:.6f}")
    
    return feature_stats, background_threshold

def parse_arduino_packet(packet_data):
    """
    Parse Arduino packet format: timestamp,sample1,sample2,...,sample200
    
    Args:
        packet_data: Raw UDP packet data (bytes)
    
    Returns:
        tuple: (timestamp, samples_array) or (None, None) if parsing fails
    """
    try:
        # Decode packet
        packet_str = packet_data.decode('utf-8').strip()
        
        # Split by comma
        parts = packet_str.split(',')
        
        if len(parts) != EXPECTED_SAMPLES + 1:  # timestamp + 200 samples
            print(f"ERROR: Expected {EXPECTED_SAMPLES + 1} values, got {len(parts)}")
            return None, None
        
        # Extract timestamp and samples
        timestamp = int(parts[0])
        samples = [float(x) for x in parts[1:]]
        
        # Convert to numpy array and reshape to [1, 200]
        samples_array = np.array(samples, dtype=np.float32).reshape(1, -1)
        
        return timestamp, samples_array
        
    except (ValueError, UnicodeDecodeError, IndexError) as e:
        print(f"ERROR parsing packet: {e}")
        return None, None

def extract_features_from_samples(samples_array):
    """
    Extract features from 200 samples using the same pipeline as training.
    
    Args:
        samples_array: numpy array of shape [1, 200]
    
    Returns:
        dict: extracted features
    """
    try:
        # Compute PSD using the same parameters as training
        freqs, psd = compute_psd(samples_array)
        
        # Extract features using the same function as training
        features = extract_features_from_sample(samples_array, freqs, psd)
        
        return features
        
    except Exception as e:
        print(f"ERROR extracting features: {e}")
        return None

def normalize_and_format_features(features, feature_stats):
    """
    Normalize features and convert to array format.
    
    Args:
        features: dict of extracted features
        feature_stats: loaded feature statistics
    
    Returns:
        tuple: (normalized_array, feature_names) or (None, None) if failed
    """
    try:
        # Normalize features using training statistics
        normalized_features = normalize_features(features, feature_stats, verbose=False)
        
        # Convert to array format for model input
        feature_array, feature_names = features_dict_to_array(normalized_features)
        
        return feature_array, feature_names
        
    except Exception as e:
        print(f"ERROR normalizing features: {e}")
        return None, None

def process_sensor_data(samples_array, timestamp, feature_stats, background_threshold):
    """
    Process sensor data through the complete feature extraction pipeline.
    
    Args:
        samples_array: numpy array of shape [1, 200]
        timestamp: Arduino timestamp
        feature_stats: loaded feature statistics
        background_threshold: background energy threshold
    
    Returns:
        dict: processing results
    """
    # Extract features
    features = extract_features_from_samples(samples_array)
    if features is None:
        return None
    
    # Normalize features
    feature_array, feature_names = normalize_and_format_features(features, feature_stats)
    if feature_array is None:
        return None
    
    # Check background energy threshold
    total_power = features['total_power']
    is_background = total_power < background_threshold
    
    # Format results
    results = {
        'timestamp': timestamp,
        'raw_features': features,
        'normalized_features': feature_array[0],  # Single sample
        'feature_names': feature_names,
        'total_power': total_power,
        'background_threshold': background_threshold,
        'is_background': is_background,
        'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return results

def print_results(results):
    """Print formatted results to console."""
    print("\n" + "="*80)
    print(f"SEISMIC FEATURE EXTRACTION - {results['processing_time']}")
    print("="*80)
    print(f"Arduino Timestamp: {results['timestamp']}")
    print(f"Total Power: {results['total_power']:.6f}")
    print(f"Background Threshold: {results['background_threshold']:.6f}")
    print(f"Background Detection: {'YES' if results['is_background'] else 'NO'}")
    
    print(f"\nRaw Features:")
    for feature_name, value in results['raw_features'].items():
        print(f"  {feature_name:20s}: {value:.6f}")
    
    print(f"\nNormalized Features:")
    for i, (feature_name, value) in enumerate(zip(results['feature_names'], results['normalized_features'])):
        print(f"  {feature_name:20s}: {value:.6f}")
    
    print("="*80)

def main():
    """Main UDP listener loop."""
    print("="*80)
    print("REAL-TIME SEISMIC FEATURE EXTRACTION SYSTEM")
    print("="*80)
    print(f"Listening on UDP port {UDP_PORT}")
    print(f"Expected format: timestamp,sample1,sample2,...,sample{EXPECTED_SAMPLES}")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    # Load statistics
    feature_stats, background_threshold = load_statistics()
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', UDP_PORT))
    
    print(f"✓ UDP server started on port {UDP_PORT}")
    print("Waiting for sensor data...")
    
    packet_count = 0
    
    try:
        while True:
            # Receive UDP packet
            data, addr = sock.recvfrom(BUFFER_SIZE)
            packet_count += 1
            
            print(f"\n📦 Received packet #{packet_count} from {addr}")
            
            # Parse packet
            timestamp, samples_array = parse_arduino_packet(data)
            # multiple samples array by 16000
            samples_array = samples_array * 16000
            if timestamp is None:
                print("❌ Failed to parse packet, skipping...")
                continue
            
            print(f"✓ Parsed {samples_array.shape[1]} samples, timestamp: {timestamp}")
            
            # Process through feature extraction pipeline
            results = process_sensor_data(
                samples_array, timestamp, feature_stats, background_threshold
            )
            
            if results is None:
                print("❌ Feature extraction failed, skipping...")
                continue
            
            # Display results
            print_results(results)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        sock.close()
        print("✓ UDP socket closed")

if __name__ == "__main__":
    main()
