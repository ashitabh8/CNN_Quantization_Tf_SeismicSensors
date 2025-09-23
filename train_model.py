#!/usr/bin/env python3
"""
Training Script for Seismic Sensor Data Classification
Uses MultiModalDataset with lazy loading and ConfigurableCNN model
"""

import os
import yaml
import tensorflow as tf
import numpy as np
from datetime import datetime
from dataset import MultiModalDataset
from models import ConfigurableCNN

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class Args:
    """Simple args class to hold configuration"""
    def __init__(self, config):
        self.dataset_config = config
        self.task = "vehicle_classification"

def normalize_batch(data, label):
    """
    Normalize input data per batch using batch normalization approach
    """
    # Simple tensor normalization for seismic data
    mean = tf.reduce_mean(data, axis=0, keepdims=True)
    std = tf.math.reduce_std(data, axis=0, keepdims=True)
    normalized_data = (data - mean) / (std + 1e-8)
    return normalized_data, label

def get_input_shape_from_dataset(dataset):
    """
    Determine input shape from the seismic data in the dataset
    """
    sample_data, _, _ = dataset[0]
    
    # Extract only the seismic data from data['shake']['seismic']
    if isinstance(sample_data, dict):
        seismic_data = sample_data['shake']['seismic']
        print(f"Extracted seismic data shape: {seismic_data.shape}")
        return seismic_data.shape
    else:
        return sample_data.shape

def extract_seismic_data(data, label):
    """
    Extract only the seismic data from data['shake']['seismic']
    """
    if isinstance(data, dict):
        # Extract only the seismic data
        seismic_data = data['shake']['seismic']
        return seismic_data, label
    else:
        return data, label

def create_datasets(config_path):
    """
    Create train, validation, and test datasets with lazy loading and 0 caching
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    args = Args(config)
    
    print("Creating datasets with lazy loading and 0 caching...")
    
    # Create dataset instances
    train_dataset = MultiModalDataset(
        args=args,
        index_file=config['vehicle_classification']['train_index_file']
    )
    
    val_dataset = MultiModalDataset(
        args=args,
        index_file=config['vehicle_classification']['val_index_file']
    )
    
    test_dataset = MultiModalDataset(
        args=args,
        index_file=config['vehicle_classification']['test_index_file']
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, config

def create_tf_datasets(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Convert to TensorFlow datasets with lazy loading, 0 caching, and batch normalization
    """
    print("Converting to TensorFlow datasets with lazy loading (cache_size=0)...")
    
    # Create TF datasets with lazy loading and 0 caching for maximum efficiency
    train_tf_dataset = train_dataset.to_tf_dataset_lazy(
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        cache_size=0  # 0 caching for efficiency as requested
    )
    
    val_tf_dataset = val_dataset.to_tf_dataset_lazy(
        batch_size=batch_size,
        shuffle_buffer_size=0,  # No shuffling for validation
        cache_size=0  # 0 caching for efficiency
    )
    
    test_tf_dataset = test_dataset.to_tf_dataset_lazy(
        batch_size=batch_size,
        shuffle_buffer_size=0,  # No shuffling for test
        cache_size=0  # 0 caching for efficiency
    )
    
    # Extract only seismic data from data['shake']['seismic']
    sample_data, _, _ = train_dataset[0]
    if isinstance(sample_data, dict):
        print("Extracting seismic data from data['shake']['seismic']...")
        train_tf_dataset = train_tf_dataset.map(extract_seismic_data, num_parallel_calls=tf.data.AUTOTUNE)
        val_tf_dataset = val_tf_dataset.map(extract_seismic_data, num_parallel_calls=tf.data.AUTOTUNE)
        test_tf_dataset = test_tf_dataset.map(extract_seismic_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply batch normalization
    print("Applying per-batch normalization...")
    train_tf_dataset = train_tf_dataset.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_tf_dataset = test_tf_dataset.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_tf_dataset, val_tf_dataset, test_tf_dataset

def create_model(input_shape, num_classes):
    """
    Create and compile the ConfigurableCNN model
    """
    print(f"Creating ConfigurableCNN model with input_shape={input_shape}, num_classes={num_classes}")
    
    # Create the CNN model
    cnn = ConfigurableCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dense_units=512
    )
    
    # Build and compile the model
    model = cnn.get_model()
    
    print("Model created successfully!")
    print("\nModel Summary:")
    cnn.summary()
    
    return cnn, model

def train_and_evaluate(model, train_dataset, val_dataset, test_dataset, epochs=2):
    """
    Train the model for specified epochs and evaluate on test set with TensorBoard logging
    """
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    # Create TensorBoard log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/seismic_training_{timestamp}"
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir {log_dir}")
    
    # Define callbacks for better training monitoring
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # Log weight histograms every epoch
            write_graph=True,  # Log the model graph
            write_images=True,  # Log model weights as images
            update_freq='epoch',  # Log metrics every epoch
            profile_batch=2,  # Profile the second batch for performance analysis
            embeddings_freq=1  # Log embeddings every epoch
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    print("=" * 60)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view training progress, run: tensorboard --logdir {log_dir}")
    print("Then open http://localhost:6006 in your browser")
    
    return history, test_results, log_dir

def main():
    """
    Main training function
    """
    print("=" * 80)
    print("SEISMIC SENSOR DATA CLASSIFICATION TRAINING")
    print("=" * 80)
    print("Configuration:")
    print("- Using MultiModalDataset with lazy loading")
    print("- Cache size: 0 (no caching for maximum efficiency)")
    print("- Per-batch normalization enabled")
    print("- Training epochs: 2")
    print("- Model: ConfigurableCNN")
    print("- Data focus: Only data['shake']['seismic'] samples")
    print("=" * 80)
    
    # Configuration
    config_path = 'dataset_config.yaml'
    batch_size = 32
    epochs = 2
    
    try:
        # Create datasets
        train_dataset, val_dataset, test_dataset, config = create_datasets(config_path)
        
        # Get input shape from first sample
        input_shape = get_input_shape_from_dataset(train_dataset)
        num_classes = config['vehicle_classification']['num_classes']
        
        print(f"\nDataset Configuration:")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {batch_size}")
        
        # Create TensorFlow datasets
        train_tf_dataset, val_tf_dataset, test_tf_dataset = create_tf_datasets(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        # Create model
        cnn, model = create_model(input_shape, num_classes)
        
        # Train and evaluate
        history, test_results, log_dir = train_and_evaluate(
            model, train_tf_dataset, val_tf_dataset, test_tf_dataset, epochs
        )
        
        # Save the trained model
        model_save_path = 'trained_seismic_model'
        print(f"\nSaving model to {model_save_path}...")
        cnn.save_for_tflite(model_save_path)

        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final test accuracy: {test_results[1]:.4f}")
        print(f"Model saved to: {model_save_path}")
        print(f"TensorBoard logs: {log_dir}")
        print("\nTo view training metrics and progress:")
        print(f"1. Run: tensorboard --logdir {log_dir}")
        print("2. Open http://localhost:6006 in your browser")

        return history, test_results, cnn, log_dir

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please check your dataset paths and configuration.")
        raise

if __name__ == "__main__":
    # Run the training
    history, test_results, trained_cnn, log_dir = main()
