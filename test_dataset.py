#!/usr/bin/env python3
"""
Test script for the TensorFlow Keras MultiModalDataset class
"""

import yaml
import tensorflow as tf
import numpy as np
from dataset import MultiModalDataset

def test_to_tf_dataset_lazy():
    """
    Test the to_tf_dataset_lazy function and print detailed summary statistics
    for 2 batches to understand dataset structure and usage.
    """
    print("=" * 60)
    print("Testing to_tf_dataset_lazy function")
    print("=" * 60)
    
    # Load dataset configuration
    try:
        with open('dataset_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Successfully loaded dataset configuration")
        print(f"  - Task: vehicle_classification")
        print(f"  - Number of classes: {config['vehicle_classification']['num_classes']}")
        print(f"  - Class names: {config['vehicle_classification']['class_names']}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Create args object
    class Args:
        def __init__(self):
            self.dataset_config = config
            self.task = "vehicle_classification"
    
    args = Args()
    
    # Test dataset creation
    try:
        train_dataset = MultiModalDataset(
            args=args,
            index_file=config['vehicle_classification']['train_index_file']
        )
        print(f"✓ Created training dataset with {len(train_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return
    
    # Test to_tf_dataset_lazy with different configurations
    test_configs = [
        {"batch_size": 4, "shuffle_buffer_size": 100, "cache_size": 50, "name": "With Caching"},
        {"batch_size": 4, "shuffle_buffer_size": 100, "cache_size": 0, "name": "Without Caching"}
    ]
    
    for config_idx, test_config in enumerate(test_configs):
        print(f"\n{'-' * 40}")
        print(f"Test Configuration {config_idx + 1}: {test_config['name']}")
        print(f"{'-' * 40}")
        
        try:
            # Create TensorFlow dataset using to_tf_dataset_lazy
            tf_dataset = train_dataset.to_tf_dataset_lazy(
                batch_size=test_config["batch_size"],
                shuffle_buffer_size=test_config["shuffle_buffer_size"],
                cache_size=test_config["cache_size"]
            )
            print(f"✓ Successfully created TF dataset with lazy loading")
            print(f"  - Batch size: {test_config['batch_size']}")
            print(f"  - Shuffle buffer size: {test_config['shuffle_buffer_size']}")
            print(f"  - Cache size: {test_config['cache_size']}")
            
            # Analyze 2 batches
            batch_count = 0
            for batch_data, batch_labels in tf_dataset.take(2):
                batch_count += 1
                print(f"\n--- Batch {batch_count} Analysis ---")
                
                # Batch-level statistics
                print(f"Batch shape: {batch_data.shape if not isinstance(batch_data, dict) else 'Dictionary structure'}")
                print(f"Labels shape: {batch_labels.shape}")
                print(f"Actual batch size: {batch_labels.shape[0]}")
                
                # Focus on the specific seismic tensor
                if isinstance(batch_data, dict) and 'shake' in batch_data and 'seismic' in batch_data['shake']:
                    seismic_tensor = batch_data['shake']['seismic']
                    print("Seismic tensor analysis:")
                    print(f"  - Shape: {seismic_tensor.shape}")
                    print(f"  - Dtype: {seismic_tensor.dtype}")
                    
                    # Calculate statistics for seismic data
                    if seismic_tensor.dtype in [tf.float32, tf.float64, tf.int32, tf.int64]:
                        mean_val = tf.reduce_mean(tf.cast(seismic_tensor, tf.float32))
                        std_val = tf.math.reduce_std(tf.cast(seismic_tensor, tf.float32))
                        min_val = tf.reduce_min(seismic_tensor)
                        max_val = tf.reduce_max(seismic_tensor)
                        print(f"  - Mean: {mean_val:.4f}")
                        print(f"  - Std: {std_val:.4f}")
                        print(f"  - Min: {min_val}")
                        print(f"  - Max: {max_val}")
                        
                        # Additional statistics per sample in batch
                        print(f"  - Per-sample means: {tf.reduce_mean(seismic_tensor, axis=list(range(1, len(seismic_tensor.shape))))}")
                else:
                    print("Warning: Expected data['shake']['seismic'] structure not found")
                    print(f"Available keys: {list(batch_data.keys()) if isinstance(batch_data, dict) else 'Not a dictionary'}")
                
                # Label statistics
                print(f"Labels dtype: {batch_labels.dtype}")
                unique_labels, _, counts = tf.unique_with_counts(batch_labels)
                print(f"Unique labels in batch: {unique_labels.numpy()}")
                print(f"Label counts: {counts.numpy()}")
                
                # Calculate label distribution
                label_mean = tf.reduce_mean(tf.cast(batch_labels, tf.float32))
                print(f"Label average: {label_mean:.4f}")
                
                # Map labels to class names if available
                if 'class_names' in config['vehicle_classification']:
                    class_names = config['vehicle_classification']['class_names']
                    print("Class distribution in batch:")
                    for label, count in zip(unique_labels.numpy(), counts.numpy()):
                        if label < len(class_names):
                            print(f"  - {class_names[label]}: {count} samples")
                        else:
                            print(f"  - Class {label}: {count} samples")
            
            print(f"\n✓ Successfully processed 2 batches from lazy dataset")
            
        except Exception as e:
            print(f"✗ Failed to test lazy dataset: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 60}")
    print("Summary: to_tf_dataset_lazy Function Analysis")
    print(f"{'=' * 60}")
    print("The to_tf_dataset_lazy function:")
    print("1. Uses a generator-based approach for memory efficiency")
    print("2. Supports optional LRU caching for frequently accessed samples")
    print("3. Handles shuffling at the index level before data loading")
    print("4. Automatically determines output signature from sample data")
    print("5. Supports both single tensor and dictionary (multi-modal) data structures")
    print("6. Applies batching and prefetching for optimal performance")
    print("\nKey parameters:")
    print("- batch_size: Controls the number of samples per batch")
    print("- shuffle_buffer_size: Controls shuffling (0 to disable)")
    print("- cache_size: Number of samples to cache in memory (0 to disable)")

def test_dataset():
    """Test the converted TensorFlow dataset class"""
    
    # Load dataset configuration
    try:
        with open('dataset_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Successfully loaded dataset configuration")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Create args object
    class Args:
        def __init__(self):
            self.dataset_config = config
            self.task = "vehicle_classification"
    
    args = Args()
    print(f"✓ Created args object for task: {args.task}")
    
    # Test dataset creation
    try:
        train_dataset = MultiModalDataset(
            args=args,
            index_file=config['vehicle_classification']['train_index_file']
        )
        print(f"✓ Created training dataset with {len(train_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return
    
    # Test individual sample access
    try:
        if len(train_dataset) > 0:
            data, label, idx = train_dataset[0]
            print(f"✓ Successfully accessed sample 0:")
            print(f"  - Data type: {type(data)}")
            if isinstance(data, dict) and "shake" in data and "seismic" in data["shake"]:
                print(f"  - Seismic data type: {type(data['shake']['seismic'])}")
            print(f"  - Label: {label}")
            print(f"  - Index: {idx}")
        else:
            print("✗ Dataset is empty")
    except Exception as e:
        print(f"✗ Failed to access individual sample: {e}")
        return

if __name__ == "__main__":
    test_to_tf_dataset_lazy()
