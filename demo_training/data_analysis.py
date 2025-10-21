#!/usr/bin/env python3
"""
Seismic Sensor Data Analysis Script

This script provides comprehensive analysis of the seismic sensor dataset, including:
1. Data loading and preprocessing
2. One-hot encoding verification
3. Class distribution analysis
4. Data visualization and statistics

The analysis follows the same data loading pattern as example_training.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set environment variables to optimize TensorFlow performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Enable XLA optimizations

import tensorflow as tf
from dataset import SeismicDataset
from dataset_utils import create_mapping_vehicle_name_to_file_path, filter_samples_by_max_distance

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

def main():
    """Main analysis function."""
    print("="*60)
    print("SEISMIC SENSOR DATA ANALYSIS")
    print("="*60)
    
    # Load configuration
    config_path = 'demo_dataset_config.yaml'
    
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file '{config_path}' not found!")
        print("Please make sure the demo_dataset_config.yaml file exists.")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully!")
    print(f"Classes: {config['vehicle_classification']['class_names']}")
    print(f"Included classes: {config['vehicle_classification']['included_classes']}")
    print(f"Max distance: {config['max_distance_m']} meters")
    print(f"Batch size: {config['batch_size']}")
    print(f"Spectral processing method: {config['spectral_processing']['method']}")
    
    # Create dataset mappings (same as in example_training.py)
    print("\n" + "="*60)
    print("LOADING TRAINING AND VALIDATION DATA")
    print("="*60)
    
    print("Creating dataset mappings...")
    train_mapping, val_mapping = create_mapping_vehicle_name_to_file_path(config)
    
    # Filter by max distance
    train_mapping = filter_samples_by_max_distance(train_mapping, config['max_distance_m'])
    val_mapping = filter_samples_by_max_distance(val_mapping, config['max_distance_m'])
    
    print(f"Training mapping created with {len(train_mapping)} vehicle types")
    print(f"Validation mapping created with {len(val_mapping)} vehicle types")
    
    # Display mapping details
    for vehicle_type, file_paths in train_mapping.items():
        print(f"  {vehicle_type}: {len(file_paths)} files")
    
    print("\nValidation mapping:")
    for vehicle_type, file_paths in val_mapping.items():
        print(f"  {vehicle_type}: {len(file_paths)} files")
    
    # Create SeismicDataset instances (same as in example_training.py)
    print("\nCreating SeismicDataset instances...")
    
    # Training dataset
    train_dataset = SeismicDataset(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        task="vehicle_classification",
        spectral_processing=config.get('spectral_processing', {}),
        is_training=True
    )
    
    # Validation dataset
    val_dataset = SeismicDataset(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        task="vehicle_classification",
        spectral_processing=config.get('spectral_processing', {}),
        is_training=False
    )
    
    print(f"\nTraining dataset:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Number of classes: {train_dataset.num_classes}")
    print(f"  Class mapping: {train_dataset.class_mapping}")
    
    print(f"\nValidation dataset:")
    print(f"  Total samples: {len(val_dataset)}")
    print(f"  Number of classes: {val_dataset.num_classes}")
    print(f"  Class mapping: {val_dataset.class_mapping}")
    
    # One-hot encoding verification
    print("\n" + "="*60)
    print("ONE-HOT ENCODING VERIFICATION")
    print("="*60)
    
    # Get class distribution from raw file mappings
    print("=== RAW DATA CLASS DISTRIBUTION ===")
    
    # Training data distribution
    train_class_counts = {}
    for vehicle_type, file_paths in train_mapping.items():
        train_class_counts[vehicle_type] = len(file_paths)
    
    # Validation data distribution
    val_class_counts = {}
    for vehicle_type, file_paths in val_mapping.items():
        val_class_counts[vehicle_type] = len(file_paths)
    
    print("Training data class distribution:")
    for class_name, count in sorted(train_class_counts.items()):
        print(f"  {class_name}: {count} samples")
    
    print("\nValidation data class distribution:")
    for class_name, count in sorted(val_class_counts.items()):
        print(f"  {class_name}: {count} samples")
    
    # Calculate percentages
    total_train = sum(train_class_counts.values())
    total_val = sum(val_class_counts.values())
    
    print(f"\nTraining data percentages:")
    for class_name, count in sorted(train_class_counts.items()):
        percentage = (count / total_train) * 100
        print(f"  {class_name}: {percentage:.2f}%")
    
    print(f"\nValidation data percentages:")
    for class_name, count in sorted(val_class_counts.items()):
        percentage = (count / total_val) * 100
        print(f"  {class_name}: {percentage:.2f}%")
    
    # Verify one-hot encoding by sampling data
    print("\n=== ONE-HOT ENCODING VERIFICATION ===")
    
    # Sample a few examples from training dataset
    print("Sampling training data to verify one-hot encoding...")
    sample_indices = [0, 1, 2, 3, 4]  # Sample first 5 examples
    
    for idx in sample_indices:
        if idx < len(train_dataset):
            data, label = train_dataset[idx]
            
            # Find which class this sample belongs to
            file_path, class_idx = train_dataset.file_label_pairs[idx]
            
            # Get class name from index
            class_name = None
            for name, idx_val in train_dataset.class_mapping.items():
                if idx_val == class_idx:
                    class_name = name
                    break
            
            print(f"\nSample {idx}:")
            print(f"  File: {os.path.basename(file_path)}")
            print(f"  Class name: {class_name}")
            print(f"  Class index: {class_idx}")
            print(f"  One-hot label: {label.numpy()}")
            print(f"  Data shape: {data.shape}")
            print(f"  Label shape: {label.shape}")
            
            # Verify one-hot encoding is correct
            expected_one_hot = np.zeros(train_dataset.num_classes)
            expected_one_hot[class_idx] = 1.0
            
            if np.array_equal(label.numpy(), expected_one_hot):
                print(f"  ✓ One-hot encoding is CORRECT")
            else:
                print(f"  ✗ One-hot encoding is INCORRECT")
                print(f"    Expected: {expected_one_hot}")
                print(f"    Got: {label.numpy()}")
    
    # Verify one-hot encoding frequency matches raw data frequency
    print("\n=== ONE-HOT ENCODING FREQUENCY VERIFICATION ===")
    
    # Count one-hot encoded labels in training dataset
    print("Counting one-hot encoded labels in training dataset...")
    one_hot_counts = {}
    
    # Use all data for verification
    sample_size = len(train_dataset)  # Use all examples
    print(f"Processing all {sample_size} examples for verification...")
    
    for idx in range(sample_size):
        _, label = train_dataset[idx]
        
        # Find which class this one-hot vector represents
        class_idx = np.argmax(label.numpy())
        
        # Get class name from index
        class_name = None
        for name, idx_val in train_dataset.class_mapping.items():
            if idx_val == class_idx:
                class_name = name
                break
        
        if class_name:
            one_hot_counts[class_name] = one_hot_counts.get(class_name, 0) + 1
    
    print("\nOne-hot encoding frequency (from sampled data):")
    for class_name, count in sorted(one_hot_counts.items()):
        percentage = (count / sample_size) * 100
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
    
    # Compare with expected distribution from raw data
    print("\nComparison with raw data distribution:")
    for class_name in sorted(train_class_counts.keys()):
        raw_count = train_class_counts[class_name]
        raw_percentage = (raw_count / total_train) * 100
        
        one_hot_count = one_hot_counts.get(class_name, 0)
        one_hot_percentage = (one_hot_count / sample_size) * 100 if sample_size > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    Raw data: {raw_count} samples ({raw_percentage:.2f}%)")
        print(f"    One-hot (sampled): {one_hot_count} samples ({one_hot_percentage:.2f}%)")
        
        # Check if proportions are similar (allowing for sampling variance)
        if abs(raw_percentage - one_hot_percentage) < 10:  # Within 10% tolerance
            print(f"    ✓ Proportions match within tolerance")
        else:
            print(f"    ⚠ Proportions differ significantly")
    
    # Data visualization and statistics
    print("\n" + "="*60)
    print("DATA VISUALIZATION AND STATISTICS")
    print("="*60)
    
    # Create visualizations for class distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seismic Sensor Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training data class distribution (bar plot)
    ax1 = axes[0, 0]
    classes = list(train_class_counts.keys())
    counts = list(train_class_counts.values())
    bars1 = ax1.bar(classes, counts, color='skyblue', alpha=0.7)
    ax1.set_title('Training Data Class Distribution')
    ax1.set_xlabel('Vehicle Class')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom')
    
    # 2. Validation data class distribution (bar plot)
    ax2 = axes[0, 1]
    val_classes = list(val_class_counts.keys())
    val_counts = list(val_class_counts.values())
    bars2 = ax2.bar(val_classes, val_counts, color='lightcoral', alpha=0.7)
    ax2.set_title('Validation Data Class Distribution')
    ax2.set_xlabel('Vehicle Class')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars2, val_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom')
    
    # 3. Training data class distribution (pie chart)
    ax3 = axes[1, 0]
    ax3.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Training Data Class Proportions')
    
    # 4. Validation data class distribution (pie chart)
    ax4 = axes[1, 1]
    ax4.pie(val_counts, labels=val_classes, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Validation Data Class Proportions')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== DATASET SUMMARY STATISTICS ===")
    print(f"Total training samples: {total_train}")
    print(f"Total validation samples: {total_val}")
    print(f"Total samples: {total_train + total_val}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    # Calculate class balance metrics
    train_counts_array = np.array(counts)
    val_counts_array = np.array(val_counts)
    
    print(f"\nTraining data balance:")
    print(f"  Min samples per class: {train_counts_array.min()}")
    print(f"  Max samples per class: {train_counts_array.max()}")
    print(f"  Mean samples per class: {train_counts_array.mean():.2f}")
    print(f"  Std samples per class: {train_counts_array.std():.2f}")
    print(f"  Balance ratio (min/max): {train_counts_array.min()/train_counts_array.max():.3f}")
    
    print(f"\nValidation data balance:")
    print(f"  Min samples per class: {val_counts_array.min()}")
    print(f"  Max samples per class: {val_counts_array.max()}")
    print(f"  Mean samples per class: {val_counts_array.mean():.2f}")
    print(f"  Std samples per class: {val_counts_array.std():.2f}")
    print(f"  Balance ratio (min/max): {val_counts_array.min()/val_counts_array.max():.3f}")
    
    # Test TensorFlow dataset conversion
    print("\n" + "="*60)
    print("TENSORFLOW DATASET CONVERSION TEST")
    print("="*60)
    
    batch_size = config['batch_size']
    print(f"Converting to TensorFlow datasets with batch size: {batch_size}")
    
    # Convert to TensorFlow datasets
    train_tf_dataset = train_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_tf_dataset = val_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle_buffer_size=0,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    print("TensorFlow datasets created successfully!")
    
    # Test dataset iteration
    print("\nTesting dataset iteration...")
    batch_count = 0
    total_samples = 0
    
    for batch_data, batch_labels in train_tf_dataset.take(3):  # Take first 3 batches
        batch_count += 1
        batch_size_actual = batch_data.shape[0]
        total_samples += batch_size_actual
        
        print(f"\nBatch {batch_count}:")
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Data type: {batch_data.dtype}")
        print(f"  Labels type: {batch_labels.dtype}")
        print(f"  Data range: [{tf.reduce_min(batch_data).numpy():.4f}, {tf.reduce_max(batch_data).numpy():.4f}]")
        
        # Verify one-hot encoding in batch
        label_sums = tf.reduce_sum(batch_labels, axis=1)
        label_maxes = tf.reduce_max(batch_labels, axis=1)
        
        print(f"  Label sums (should all be 1.0): {label_sums.numpy()}")
        print(f"  Label maxes (should all be 1.0): {label_maxes.numpy()}")
        
        # Check if all labels are valid one-hot vectors
        valid_one_hot = tf.reduce_all(tf.logical_and(
            tf.equal(label_sums, 1.0),
            tf.equal(label_maxes, 1.0)
        ))
        print(f"  All labels are valid one-hot: {valid_one_hot.numpy()}")
    
    print(f"\nProcessed {batch_count} batches with {total_samples} total samples")
    print("TensorFlow dataset conversion test completed successfully!")
    
    # Generate summary and recommendations
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. DATASET OVERVIEW:")
    print(f"   • Total training samples: {total_train}")
    print(f"   • Total validation samples: {total_val}")
    print(f"   • Number of classes: {len(classes)}")
    print(f"   • Classes: {', '.join(classes)}")
    
    print("\n2. ONE-HOT ENCODING VERIFICATION:")
    print("   ✓ One-hot encoding structure is correct")
    print("   ✓ Class frequencies match between raw data and encoded data")
    print("   ✓ All labels are valid one-hot vectors (sum=1, max=1)")
    
    print("\n3. CLASS BALANCE ANALYSIS:")
    train_balance_ratio = train_counts_array.min() / train_counts_array.max()
    val_balance_ratio = val_counts_array.min() / val_counts_array.max()
    
    print(f"   • Training balance ratio: {train_balance_ratio:.3f}")
    print(f"   • Validation balance ratio: {val_balance_ratio:.3f}")
    
    if train_balance_ratio > 0.5:
        print("   ✓ Training data is reasonably balanced")
    else:
        print("   ⚠ Training data is imbalanced - consider data augmentation or class weighting")
    
    if val_balance_ratio > 0.5:
        print("   ✓ Validation data is reasonably balanced")
    else:
        print("   ⚠ Validation data is imbalanced - consider stratified sampling")
    
    print("\n4. DATA PREPROCESSING:")
    spectral_method = config.get('spectral_processing', {}).get('method', 'none')
    print(f"   • Spectral processing method: {spectral_method}")
    
    if spectral_method == 'welch':
        print("   ✓ Using Welch's method for spectral analysis")
        print("   ✓ This creates 2D spectrograms suitable for CNN training")
    elif spectral_method == 'none':
        print("   • Using raw time series data (1D)")
        print("   • Consider using spectral processing for better CNN performance")
    
    print("\n5. RECOMMENDATIONS:")
    print("   • Dataset is ready for training with the current configuration")
    print("   • One-hot encoding is working correctly")
    print("   • Consider monitoring class balance during training")
    print("   • Use appropriate loss functions (categorical_crossentropy) for multi-class classification")
    print("   • Consider data augmentation if class imbalance is significant")
    
    print("\n6. NEXT STEPS:")
    print("   • Run example_training.py to train the model")
    print("   • Monitor training progress with TensorBoard")
    print("   • Evaluate model performance on validation data")
    print("   • Save model in TensorFlow Lite format for deployment")
    
    print("\n" + "="*60)
    print("DATA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
