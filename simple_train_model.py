#!/usr/bin/env python3
"""
Simple Training Script for Seismic Sensor Data Classification
Uses MultiModalDataset with lazy loading and a simple CNN model
"""

import os
import yaml
import tensorflow as tf
import numpy as np
from datetime import datetime
from dataset import MultiModalDataset
from models import create_deep_efficient_cnn, create_simple_cnn, print_model_info, create_resnet_style_cnn, print_resnet_model_info, deep_cnn_large, create_ultra_lightweight_cnn
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
from experiment_utils import (
    generate_experiment_id, create_experiment_directory, save_experiment_metadata,
    save_model_summary, log_experiment_metadata_to_tensorboard, get_experiment_paths,
    get_model_type_name
)

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
    Determine input shape from the seismic data in the dataset.
    This function accounts for spectral processing (FFT, Welch) that may change the shape.
    
    Args:
        dataset: MultiModalDataset instance
        
    Returns:
        tuple: The actual input shape after any spectral processing
    """
    # Get a sample from the dataset to determine the actual processed shape
    sample_data, _, _ = dataset[0]
    
    # Extract only the seismic data from data['shake']['seismic']
    if isinstance(sample_data, dict):
        seismic_data = sample_data['shake']['seismic']
        print(f"Extracted seismic data shape: {seismic_data.shape}")
        return seismic_data.shape
    else:
        print(f"Direct seismic data shape: {sample_data.shape}")
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


class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to compute and log detailed metrics including F1 score, recall, 
    confusion matrix, and per-class performance to TensorBoard
    """
    
    def __init__(self, validation_data, num_classes, class_names=None, log_dir=None):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.log_dir = log_dir
        
        # Create file writer for custom metrics
        if log_dir:
            # Handle both old and new log directory structures
            if 'detailed_metrics' in log_dir:
                detailed_metrics_dir = log_dir
            else:
                detailed_metrics_dir = os.path.join(log_dir, 'detailed_metrics')
            self.file_writer = tf.summary.create_file_writer(detailed_metrics_dir)
    
    def plot_confusion_matrix(self, cm, epoch):
        """Create and return confusion matrix plot as image"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to tensor
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        plt.close()
        buf.close()
        
        return image
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute detailed metrics at the end of each epoch"""
        if not hasattr(self, 'file_writer'):
            return
            
        # Get predictions on validation data
        y_true = []
        y_pred = []
        
        print(f"\nComputing detailed metrics for epoch {epoch + 1}...")
        
        for batch_x, batch_y in self.validation_data:
            # Get predictions
            predictions = self.model.predict(batch_x, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Handle one-hot encoded labels
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                # One-hot encoded labels - convert to class indices
                true_classes = np.argmax(batch_y.numpy(), axis=1)
                y_true.extend(true_classes)
            else:
                # Regular integer labels
                y_true.extend(batch_y.numpy())
            y_pred.extend(pred_classes)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute metrics
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class F1 and recall scores
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Log to TensorBoard
        with self.file_writer.as_default():
            # Overall metrics
            tf.summary.scalar('f1_score/macro', f1_macro, step=epoch)
            tf.summary.scalar('f1_score/micro', f1_micro, step=epoch)
            tf.summary.scalar('f1_score/weighted', f1_weighted, step=epoch)
            
            tf.summary.scalar('recall/macro', recall_macro, step=epoch)
            tf.summary.scalar('recall/micro', recall_micro, step=epoch)
            tf.summary.scalar('recall/weighted', recall_weighted, step=epoch)
            
            # Per-class metrics
            for i, (class_name, f1, recall) in enumerate(zip(self.class_names, f1_per_class, recall_per_class)):
                tf.summary.scalar(f'f1_score_per_class/{class_name}', f1, step=epoch)
                tf.summary.scalar(f'recall_per_class/{class_name}', recall, step=epoch)
                
                # Class support (number of samples)
                class_support = np.sum(y_true == i)
                tf.summary.scalar(f'class_support/{class_name}', class_support, step=epoch)
            
            # Confusion matrix as image
            cm_image = self.plot_confusion_matrix(cm, epoch)
            tf.summary.image('confusion_matrix', cm_image, step=epoch)
            
            self.file_writer.flush()
        
        # Print summary of top performing classes
        class_performance = list(zip(self.class_names, f1_per_class, recall_per_class))
        class_performance.sort(key=lambda x: x[1], reverse=True)  # Sort by F1 score
        
        print(f"\nTop 3 performing classes by F1 score (Epoch {epoch + 1}):")
        print("-" * 60)
        for i, (class_name, f1, recall) in enumerate(class_performance[:3]):
            print(f"{i+1}. {class_name}: F1={f1:.4f}, Recall={recall:.4f}")
        
        print(f"\nOverall Metrics (Epoch {epoch + 1}):")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"Recall (macro): {recall_macro:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")


def log_test_results_to_tensorboard(model, test_dataset, num_classes, class_names, log_dir, test_results):
    """
    Log test evaluation results to TensorBoard with detailed metrics
    """
    print("\nComputing detailed test metrics for TensorBoard logging...")
    
    # Create file writer for test metrics
    # Handle both old and new log directory structures
    if 'test_metrics' in log_dir:
        test_log_dir = log_dir
    else:
        test_log_dir = os.path.join(log_dir, 'test_metrics')
    file_writer = tf.summary.create_file_writer(test_log_dir)
    
    # Get predictions on test data
    y_true = []
    y_pred = []
    
    for batch_x, batch_y in test_dataset:
        # Get predictions
        predictions = model.predict(batch_x, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Handle one-hot encoded labels
        if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
            # One-hot encoded labels - convert to class indices
            true_classes = np.argmax(batch_y.numpy(), axis=1)
            y_true.extend(true_classes)
        else:
            # Regular integer labels
            y_true.extend(batch_y.numpy())
        y_pred.extend(pred_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute detailed metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1 and recall scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Log to TensorBoard
    with file_writer.as_default():
        # Basic test metrics
        tf.summary.scalar('test_loss', test_results[0], step=0)
        tf.summary.scalar('test_accuracy', test_results[1], step=0)
        
        # Overall detailed metrics
        tf.summary.scalar('test_f1_score/macro', f1_macro, step=0)
        tf.summary.scalar('test_f1_score/micro', f1_micro, step=0)
        tf.summary.scalar('test_f1_score/weighted', f1_weighted, step=0)
        
        tf.summary.scalar('test_recall/macro', recall_macro, step=0)
        tf.summary.scalar('test_recall/micro', recall_micro, step=0)
        tf.summary.scalar('test_recall/weighted', recall_weighted, step=0)
        
        # Per-class metrics
        for i, (class_name, f1, recall) in enumerate(zip(class_names, f1_per_class, recall_per_class)):
            tf.summary.scalar(f'test_f1_score_per_class/{class_name}', f1, step=0)
            tf.summary.scalar(f'test_recall_per_class/{class_name}', recall, step=0)
            
            # Class support (number of samples)
            class_support = np.sum(y_true == i)
            tf.summary.scalar(f'test_class_support/{class_name}', class_support, step=0)
        
        # Confusion matrix as image
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to tensor
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        plt.close()
        buf.close()
        
        tf.summary.image('test_confusion_matrix', image, step=0)
        
        file_writer.flush()
    
    # Print detailed test results
    print(f"\nDetailed Test Results:")
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test F1 Score (macro): {f1_macro:.4f}")
    print(f"Test F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Test Recall (macro): {recall_macro:.4f}")
    print(f"Test Recall (weighted): {recall_weighted:.4f}")
    
    # Print top performing classes
    class_performance = list(zip(class_names, f1_per_class, recall_per_class))
    class_performance.sort(key=lambda x: x[1], reverse=True)  # Sort by F1 score
    
    print(f"\nTop 3 performing classes on test set by F1 score:")
    print("-" * 60)
    for i, (class_name, f1, recall) in enumerate(class_performance[:3]):
        print(f"{i+1}. {class_name}: F1={f1:.4f}, Recall={recall:.4f}")
    
    print(f"\nTest metrics logged to TensorBoard at: {test_log_dir}")
    
    return {
        'test_loss': test_results[0],
        'test_accuracy': test_results[1],
        'test_f1_macro': f1_macro,
        'test_f1_weighted': f1_weighted,
        'test_recall_macro': recall_macro,
        'test_recall_weighted': recall_weighted,
        'confusion_matrix': cm,
        'class_performance': class_performance
    }


# Model functions moved to models.py

def train_and_evaluate(model, train_dataset, val_dataset, test_dataset, num_classes, epochs=2, experiment_id=None, experiment_dir=None):
    """
    Train the model for specified epochs and evaluate on test set with TensorBoard logging
    """
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    # Use experiment-based log directory if provided, otherwise fallback to timestamp
    if experiment_dir:
        log_dir = os.path.join(experiment_dir, 'logs', 'training')
        print(f"Using experiment-based logging: {experiment_id}")
    else:
        # Fallback to old system
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/seismic_training_{timestamp}"
        print(f"Using fallback logging system")
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir {os.path.dirname(log_dir)}")
    
    # Create class names (you can customize these based on your actual class labels)
    class_names = [f'Vehicle_Class_{i}' for i in range(num_classes)]
    
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
        DetailedMetricsCallback(
            validation_data=val_dataset,
            num_classes=num_classes,
            class_names=class_names,
            log_dir=os.path.dirname(log_dir) if experiment_dir else log_dir
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
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
    # breakpoint()
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
    
    print(f"\nBasic Test Results:")
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    
    # Log detailed test results to TensorBoard
    test_log_dir = os.path.join(experiment_dir, 'logs', 'test_metrics') if experiment_dir else log_dir
    detailed_test_results = log_test_results_to_tensorboard(
        model, test_dataset, num_classes, class_names, test_log_dir, test_results
    )
    
    if experiment_dir:
        print(f"\nTensorBoard logs saved to: {os.path.join(experiment_dir, 'logs')}")
        print(f"To view training progress, run: tensorboard --logdir {os.path.join(experiment_dir, 'logs')}")
    else:
        print(f"\nTensorBoard logs saved to: {log_dir}")
        print(f"To view training progress, run: tensorboard --logdir {log_dir}")
    print("Then open http://localhost:6006 in your browser")
    print("Test metrics are available under the 'test_metrics' tab in TensorBoard")
    
    return history, test_results, log_dir, detailed_test_results

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
    
    print("- Model: Simple CNN")
    print("- Data focus: Only data['shake']['seismic'] samples")
    print("=" * 80)
    
    # Configuration
    config_path = 'dataset_config.yaml'
    batch_size = 32
    epochs = 50  # Reduced for testing
    print("- Training epochs: ", epochs)
    
    try:
        # Create datasets
        train_dataset, val_dataset, test_dataset, config = create_datasets(config_path)
        
        # Get input shape from first sample
        input_shape = get_input_shape_from_dataset(train_dataset)
        # breakpoint()
        # Get num_classes from dataset (handles class mapping automatically)
        num_classes = train_dataset.num_classes_for_training
        
        print(f"\nDataset Configuration:")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {batch_size}")
        
        # Create TensorFlow datasets
        train_tf_dataset, val_tf_dataset, test_tf_dataset = create_tf_datasets(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        # Create model
        # model = create_simple_cnn(input_shape, num_classes)
        # model = create_deep_efficient_cnn(input_shape, num_classes)
        model = create_ultra_lightweight_cnn(input_shape, num_classes)
        # model = create_resnet_style_cnn(input_shape, num_classes)
        # model = deep_cnn_large(input_shape, num_classes)
        # Build the model by specifying input shape
        model.build(input_shape=(None,) + input_shape)
        print_model_info(model)
        # exit()
        # print_resnet_model_info(model)
        
        # === EXPERIMENT TRACKING SETUP ===
        # Get model type from the actual model (will be set after model creation)
        model_type = get_model_type_name(model)
        
        # Prepare configuration for experiment tracking
        training_config = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': 0.001,  # Default from model compilation
            'input_shape': input_shape,
            'num_classes': num_classes,
            'model_type': model_type,  # Use actual model type
            'dataset_config': config_path
        }
        
        # Generate unique experiment ID
        experiment_id = generate_experiment_id(model, training_config, input_shape, num_classes)
        print(f"\nExperiment ID: {experiment_id}")
        
        # Create experiment directory structure
        experiment_dir = create_experiment_directory(experiment_id)
        print(f"Experiment directory: {experiment_dir}")
        
        # Prepare dataset info
        dataset_info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'input_shape': input_shape,
            'num_classes': num_classes
        }
        
        # Save experiment metadata
        metadata_path = save_experiment_metadata(
            experiment_dir, model, training_config, input_shape, num_classes, dataset_info
        )
        print(f"Metadata saved to: {metadata_path}")
        
        # Save model summary
        summary_path = save_model_summary(experiment_dir, model)
        print(f"Model summary saved to: {summary_path}")
        
        # Train and evaluate
        history, test_results, log_dir, detailed_test_results = train_and_evaluate(
            model, train_tf_dataset, val_tf_dataset, test_tf_dataset, num_classes, epochs, experiment_id, experiment_dir
        )
        
        # Save the trained model using experiment-based path
        model_save_path = os.path.join(experiment_dir, 'model')
        print(f"\nSaving model to {model_save_path}...")
        model.export(model_save_path)
        
        # Also save in Keras v3 and legacy H5 formats for easier downstream analysis
        keras_model_path = os.path.join(experiment_dir, 'model.keras')
        h5_model_path = os.path.join(experiment_dir, 'model.h5')
        try:
            print(f"Saving Keras model to {keras_model_path}...")
            model.save(keras_model_path)
        except Exception as e:
            print(f"Warning: Failed to save .keras format: {e}")
        
        try:
            print(f"Saving H5 model to {h5_model_path}...")
            model.save(h5_model_path)
        except Exception as e:
            print(f"Warning: Failed to save .h5 format: {e}")
        
        # Log experiment metadata to TensorBoard
        metadata = {
            'model_info': {
                'type': model_type,  # Use actual model type
                'name': getattr(model, 'name', 'unnamed'),
                'input_shape': input_shape,
                'num_classes': num_classes,
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            },
            'training_config': training_config,
            'dataset_info': dataset_info
        }
        
        # Log metadata to TensorBoard
        log_experiment_metadata_to_tensorboard(os.path.join(experiment_dir, 'logs'), metadata)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final test accuracy: {test_results[1]:.4f}")
        print(f"Model saved to: {model_save_path}")
        print(f"Keras .keras file: {keras_model_path}")
        print(f"Legacy .h5 file: {h5_model_path}")
        print(f"TensorBoard logs: {os.path.join(experiment_dir, 'logs')}")
        print(f"Experiment ID: {experiment_id}")
        print("\nTo view training metrics and progress:")
        print(f"1. Run: tensorboard --logdir {os.path.join(experiment_dir, 'logs')}")
        print("2. Open http://localhost:6006 in your browser")
        print("\nExperiment Summary:")
        print(f"- Experiment Directory: {experiment_dir}")
        print(f"- Model Type: {model_type}")
        print(f"- Input Shape: {input_shape}")
        print(f"- Classes: {num_classes}")
        print(f"- Parameters: {model.count_params():,}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch Size: {batch_size}")
        
        return history, test_results, model, experiment_dir, detailed_test_results, experiment_id
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please check your dataset paths and configuration.")
        raise

if __name__ == "__main__":
    # Run the training
    history, test_results, trained_model, experiment_dir, detailed_test_results, experiment_id = main()
