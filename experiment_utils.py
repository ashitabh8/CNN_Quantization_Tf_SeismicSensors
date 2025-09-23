#!/usr/bin/env python3
"""
Experiment Management Utilities for Seismic Sensor Data Classification
Provides organized model saving and TensorBoard logging with consistent naming
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import tensorflow as tf


def get_model_type_name(model) -> str:
    """
    Extract model type name from model object or function name.
    
    Args:
        model: TensorFlow model object
        
    Returns:
        str: Model type name (e.g., 'ultra_lightweight_cnn', 'simple_cnn')
    """
    # Try to get model name from the custom _name attribute first
    if hasattr(model, '_name') and model._name:
        return model._name.lower().replace('_', '_')
    
    # Try to get model name from the model itself
    if hasattr(model, 'name') and model.name:
        return model.name.lower().replace('_', '_')
    
    # Try to infer from model architecture
    if hasattr(model, 'layers'):
        layer_count = len(model.layers)
        param_count = model.count_params()
        
        if param_count < 50000:  # Ultra lightweight
            return 'ultra_lightweight_cnn'
        elif param_count < 200000:  # Simple
            return 'simple_cnn'
        elif param_count < 1000000:  # Deep efficient
            return 'deep_efficient_cnn'
        elif layer_count > 20:  # ResNet style
            return 'resnet_style_cnn'
        else:  # Large
            return 'deep_cnn_large'
    
    # Default fallback
    return 'unknown_cnn'


def generate_config_hash(config: Dict[str, Any]) -> str:
    """
    Generate a short hash from configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        str: 8-character hash of key configuration parameters
    """
    # Key parameters that affect model behavior
    key_params = {
        'batch_size': config.get('batch_size', 32),
        'epochs': config.get('epochs', 50),
        'learning_rate': config.get('learning_rate', 0.001),
        'input_shape': str(config.get('input_shape', 'unknown')),
        'num_classes': config.get('num_classes', 'unknown'),
        'model_type': config.get('model_type', 'unknown'),
        'dataset_config': config.get('dataset_config', 'default')
    }
    
    # Create hash from sorted key-value pairs
    config_str = json.dumps(key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return config_hash


def generate_experiment_id(model, config: Dict[str, Any], input_shape: tuple, num_classes: int) -> str:
    """
    Generate unique experiment identifier.
    
    Args:
        model: TensorFlow model object
        config: Configuration dictionary
        input_shape: Input tensor shape
        num_classes: Number of output classes
        
    Returns:
        str: Unique experiment ID
    """
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model type
    model_type = get_model_type_name(model)
    
    # Get batch size
    batch_size = config.get('batch_size', 32)
    
    # Generate config hash
    config_hash = generate_config_hash(config)
    
    # Create experiment ID
    experiment_id = f"{timestamp}_{model_type}_{num_classes}classes_{batch_size}batch_{config_hash}"
    
    return experiment_id


def create_experiment_directory(experiment_id: str, base_dir: str = "experiments") -> str:
    """
    Create organized directory structure for experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        base_dir: Base directory for all experiments
        
    Returns:
        str: Path to experiment directory
    """
    # Create base experiments directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create experiment directory
    experiment_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['model', 'logs', 'logs/training', 'logs/detailed_metrics', 'logs/test_metrics']
    for subdir in subdirs:
        subdir_path = os.path.join(experiment_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
    
    return experiment_dir


def save_experiment_metadata(experiment_dir: str, model, config: Dict[str, Any], 
                           input_shape: tuple, num_classes: int, 
                           dataset_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Save experiment metadata to JSON file.
    
    Args:
        experiment_dir: Path to experiment directory
        model: TensorFlow model object
        config: Configuration dictionary
        input_shape: Input tensor shape
        num_classes: Number of output classes
        dataset_info: Optional dataset information
        
    Returns:
        str: Path to metadata file
    """
    # Prepare metadata
    metadata = {
        'experiment_id': os.path.basename(experiment_dir),
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'type': get_model_type_name(model),
            'name': getattr(model, 'name', 'unnamed'),
            'input_shape': input_shape,
            'num_classes': num_classes,
            'total_params': model.count_params(),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
        },
        'training_config': config,
        'dataset_info': dataset_info or {},
        'system_info': {
            'tensorflow_version': tf.__version__,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(experiment_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata_path


def save_model_summary(experiment_dir: str, model) -> str:
    """
    Save model summary to text file.
    
    Args:
        experiment_dir: Path to experiment directory
        model: TensorFlow model object
        
    Returns:
        str: Path to summary file
    """
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    model_summary = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save summary
    summary_path = os.path.join(experiment_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(model_summary)
    
    return summary_path


def log_experiment_metadata_to_tensorboard(log_dir: str, metadata: Dict[str, Any]) -> None:
    """
    Log experiment metadata to TensorBoard.
    
    Args:
        log_dir: TensorBoard log directory
        metadata: Experiment metadata dictionary
    """
    # Create file writer for metadata
    metadata_log_dir = os.path.join(log_dir, 'experiment_metadata')
    file_writer = tf.summary.create_file_writer(metadata_log_dir)
    
    with file_writer.as_default():
        # Log model info
        tf.summary.text('model/type', metadata['model_info']['type'], step=0)
        tf.summary.text('model/name', metadata['model_info']['name'], step=0)
        tf.summary.scalar('model/total_params', metadata['model_info']['total_params'], step=0)
        tf.summary.scalar('model/trainable_params', metadata['model_info']['trainable_params'], step=0)
        tf.summary.scalar('model/num_classes', metadata['model_info']['num_classes'], step=0)
        
        # Log training config
        for key, value in metadata['training_config'].items():
            if isinstance(value, (int, float)):
                tf.summary.scalar(f'config/{key}', value, step=0)
            else:
                tf.summary.text(f'config/{key}', str(value), step=0)
        
        # Log dataset info
        for key, value in metadata['dataset_info'].items():
            if isinstance(value, (int, float)):
                tf.summary.scalar(f'dataset/{key}', value, step=0)
            else:
                tf.summary.text(f'dataset/{key}', str(value), step=0)
        
        file_writer.flush()


def get_experiment_paths(experiment_id: str, base_dir: str = "experiments") -> Dict[str, str]:
    """
    Get all paths for an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        base_dir: Base directory for all experiments
        
    Returns:
        Dict[str, str]: Dictionary with all experiment paths
    """
    experiment_dir = os.path.join(base_dir, experiment_id)
    
    return {
        'experiment_dir': experiment_dir,
        'model_dir': os.path.join(experiment_dir, 'model'),
        'logs_dir': os.path.join(experiment_dir, 'logs'),
        'training_logs': os.path.join(experiment_dir, 'logs', 'training'),
        'detailed_metrics_logs': os.path.join(experiment_dir, 'logs', 'detailed_metrics'),
        'test_metrics_logs': os.path.join(experiment_dir, 'logs', 'test_metrics'),
        'metadata_file': os.path.join(experiment_dir, 'metadata.json'),
        'summary_file': os.path.join(experiment_dir, 'model_summary.txt')
    }


def list_experiments(base_dir: str = "experiments") -> list:
    """
    List all available experiments.
    
    Args:
        base_dir: Base directory for all experiments
        
    Returns:
        list: List of experiment IDs
    """
    if not os.path.exists(base_dir):
        return []
    
    experiments = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'metadata.json')):
            experiments.append(item)
    
    return sorted(experiments, reverse=True)  # Most recent first


def load_experiment_metadata(experiment_id: str, base_dir: str = "experiments") -> Dict[str, Any]:
    """
    Load experiment metadata.
    
    Args:
        experiment_id: Unique experiment identifier
        base_dir: Base directory for all experiments
        
    Returns:
        Dict[str, Any]: Experiment metadata
    """
    metadata_path = os.path.join(base_dir, experiment_id, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def print_experiment_summary(experiment_id: str, base_dir: str = "experiments") -> None:
    """
    Print a summary of an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        base_dir: Base directory for all experiments
    """
    try:
        metadata = load_experiment_metadata(experiment_id, base_dir)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {experiment_id}")
        print(f"{'='*80}")
        
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Model Type: {metadata['model_info']['type']}")
        print(f"Input Shape: {metadata['model_info']['input_shape']}")
        print(f"Number of Classes: {metadata['model_info']['num_classes']}")
        print(f"Total Parameters: {int(metadata['model_info']['total_params']):,}")
        print(f"Trainable Parameters: {int(metadata['model_info']['trainable_params']):,}")
        
        print(f"\nTraining Configuration:")
        for key, value in metadata['training_config'].items():
            print(f"  {key}: {value}")
        
        if metadata['dataset_info']:
            print(f"\nDataset Information:")
            for key, value in metadata['dataset_info'].items():
                print(f"  {key}: {value}")
        
        print(f"\nTensorBoard Command:")
        logs_dir = os.path.join(base_dir, experiment_id, 'logs')
        print(f"  tensorboard --logdir {logs_dir}")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error loading experiment {experiment_id}: {e}")


if __name__ == "__main__":
    # Example usage
    print("Experiment Management Utilities")
    print("Available functions:")
    print("- generate_experiment_id()")
    print("- create_experiment_directory()")
    print("- save_experiment_metadata()")
    print("- list_experiments()")
    print("- load_experiment_metadata()")
    print("- print_experiment_summary()")
