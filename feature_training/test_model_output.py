#!/usr/bin/env python3
"""
Test script to extract raw features from CSV samples for C implementation comparison.

This script:
1. Loads raw samples from CSV file
2. Extracts raw features (psd_35hz, psd_40hz, psd_45hz, total_power)
3. Prints feature values for each sample
4. Saves raw features to CSV for C implementation comparison
"""

import numpy as np
import pandas as pd

# Import our utilities
from feature_utils import extract_all_features_from_data




def process_raw_samples_csv(csv_path):
    """Process raw samples from CSV and output raw features only."""
    print("\n" + "="*80)
    print("PROCESSING RAW SAMPLES FROM CSV")
    print("="*80)
    
    # Read CSV file
    print(f"Reading raw samples from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Extract signal data
    signal_columns = [col for col in df.columns if col.startswith('signal_')]
    print(f"Signal columns: {len(signal_columns)}")
    
    # Prepare data for feature extraction
    data_dict = {}
    for idx, row in df.iterrows():
        class_name = row['class']
        signal_data = row[signal_columns].values.astype(np.float32)
        
        if class_name not in data_dict:
            data_dict[class_name] = []
        data_dict[class_name].append(signal_data)
    
    print("Data structure:")
    for class_name, signals in data_dict.items():
        print(f"  {class_name}: {len(signals)} samples")
    
    # Extract features from all samples
    print("\nExtracting features from CSV samples...")
    raw_features = extract_all_features_from_data(data_dict, verbose=True)
    
    # Get feature names (excluding vehicle_labels)
    feature_names = [name for name in raw_features.keys() if name != 'vehicle_labels']
    
    print(f"\n✓ Features extracted:")
    print(f"  Feature names: {feature_names}")
    print(f"  Total features: {len(feature_names)}")
    
    # Process each sample
    raw_features_data = []
    
    print("\n" + "="*80)
    print("PROCESSING EACH SAMPLE")
    print("="*80)
    
    for idx, row in df.iterrows():
        sample_index = idx + 1
        class_name = row['class']
        
        print(f"\n--- Sample {sample_index} ({class_name}) ---")
        
        # Get raw features for this sample
        raw_feature_values = []
        for feature_name in feature_names:
            raw_feature_values.append(raw_features[feature_name][idx])
        
        print("Raw Features:")
        for i, feature_name in enumerate(feature_names):
            print(f"  {feature_name}: {raw_feature_values[i]:.6f}")
        
        # Save raw features to CSV
        raw_features_row = {
            'sample_index': sample_index,
            'class': class_name,
            'psd_35hz': raw_feature_values[feature_names.index('psd_35hz')],
            'psd_40hz': raw_feature_values[feature_names.index('psd_40hz')],
            'psd_45hz': raw_feature_values[feature_names.index('psd_45hz')],
            'total_power': raw_feature_values[feature_names.index('total_power')]
        }
        raw_features_data.append(raw_features_row)
    
    # Save to CSV file
    print("\n" + "="*80)
    print("SAVING CSV FILES")
    print("="*80)
    
    # Save raw features
    raw_features_df = pd.DataFrame(raw_features_data)
    raw_features_csv_path = 'python_raw_features.csv'
    raw_features_df.to_csv(raw_features_csv_path, index=False)
    print(f"✓ Raw features saved to: {raw_features_csv_path}")
    
    print(f"\nSummary:")
    print(f"  Total samples processed: {len(raw_features_data)}")
    print(f"  Features per sample: 4 (psd_35hz, psd_40hz, psd_45hz, total_power)")
    print(f"  Classes: {list(data_dict.keys())}")
    
    return raw_features_data


def main():
    """Main test pipeline - simplified to focus on raw feature extraction."""
    print("="*80)
    print("RAW FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Process raw samples from CSV
    raw_samples_csv_path = '/home/misra8/CNN_Quantization_Tf_SeismicSensors/feature_training/raw_samples.csv'
    features_data = process_raw_samples_csv(raw_samples_csv_path)
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("CSV files generated:")
    print(f"  python_raw_features.csv")
    print("Use this file to compare with your C implementation outputs.")


if __name__ == "__main__":
    main()
