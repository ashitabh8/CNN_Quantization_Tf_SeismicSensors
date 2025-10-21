import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


def calculate_background_energy_threshold(train_features_dict, config):
    """
    Compute energy threshold from training background samples using configurable method.
    
    Args:
        train_features_dict: dict with 'total_power' and 'vehicle_labels' keys
        config: Configuration dict with background_filtering settings
    
    Returns:
        float: Energy threshold for background classification
    """
    # Get background samples only
    background_mask = train_features_dict['vehicle_labels'] == 'background'
    background_energies = train_features_dict['total_power'][background_mask]
    
    if len(background_energies) == 0:
        raise ValueError("No background samples found in training data!")
    
    # Get filtering method from config
    method = config.get('background_filtering', {}).get('method', 'median')
    
    if method == 'percentile':
        percentile = config.get('background_filtering', {}).get('percentile', 25)
        threshold = np.percentile(background_energies, percentile)
        print(f"Background energy threshold ({percentile}th percentile): {threshold:.6f}")
    else:  # default to median
        threshold = np.median(background_energies)
        print(f"Background energy threshold (median): {threshold:.6f}")
    
    print(f"Background samples used: {len(background_energies)}")
    print(f"Threshold method: {method}")
    
    return float(threshold)


def create_dense_model(input_dim, num_classes, config):
    """
    Build deeper TF dense network with more capacity while staying under 600KB.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        config: Model configuration dict
    
    Returns:
        tf.keras.Model: Compiled model
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(
        config['model']['hidden_layers'][0], 
        activation=config['model']['activation'],
        input_shape=(input_dim,),
        name='dense_1'
    ))
    model.add(Dropout(config['model']['dropout_rate'], name='dropout_1'))
    
    # Hidden layers (dynamic based on config)
    for i, layer_size in enumerate(config['model']['hidden_layers'][1:], 2):
        model.add(Dense(
            layer_size, 
            activation=config['model']['activation'],
            name=f'dense_{i}'
        ))
        model.add(Dropout(config['model']['dropout_rate'], name=f'dropout_{i}'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    return model


def compile_model(model, config):
    """
    Compile model with Adam optimizer and categorical crossentropy.
    
    Args:
        model: tf.keras.Model
        config: Training configuration dict
    
    Returns:
        tf.keras.Model: Compiled model
    """
    optimizer = Adam(learning_rate=config['training']['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_size_kb(model):
    """
    Calculate model memory consumption in KB.
    
    Args:
        model: tf.keras.Model
    
    Returns:
        float: Model size in KB
    """
    # Get total number of parameters
    total_params = model.count_params()
    
    # Estimate size (assuming float32 = 4 bytes per parameter)
    size_bytes = total_params * 4
    size_kb = size_bytes / 1024
    
    return size_kb


def train_model(model, X_train, y_train, X_val, y_val, config, experiment_dir):
    """
    Training loop with callbacks.
    
    Args:
        model: Compiled tf.keras.Model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Training configuration
        experiment_dir: Directory to save results
    
    Returns:
        tf.keras.callbacks.History: Training history
    """
    # Create callbacks
    callbacks = []
    
    # Model checkpoint
    model_path = os.path.join(experiment_dir, 'models', 'best_model.keras')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor=config['output']['monitor'],
        save_best_only=config['output']['save_best_only'],
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config['output']['monitor'],
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # CSV logger
    csv_path = os.path.join(experiment_dir, 'logs', 'training_history.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    csv_logger = CSVLogger(csv_path, append=False)
    callbacks.append(csv_logger)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_training_history(history, save_path):
    """
    Plot loss and accuracy curves.
    
    Args:
        history: tf.keras.callbacks.History object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved to: {save_path}")


def save_training_config(config, experiment_dir):
    """
    Save training configuration to experiment directory.
    
    Args:
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
    """
    config_path = os.path.join(experiment_dir, 'logs', 'training_config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Training config saved to: {config_path}")


def save_model_info(model, experiment_dir):
    """
    Save model information and parameters.
    
    Args:
        model: tf.keras.Model
        experiment_dir: Path to experiment directory
    """
    # Model summary
    summary_path = os.path.join(experiment_dir, 'logs', 'model_summary.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Model parameters
    params_path = os.path.join(experiment_dir, 'logs', 'model_parameters.json')
    
    model_info = {
        'total_params': model.count_params(),
        'size_kb': get_model_size_kb(model),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    with open(params_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✓ Model info saved to: {summary_path}")
    print(f"✓ Model parameters saved to: {params_path}")
    print(f"  Model size: {model_info['size_kb']:.2f} KB")
    print(f"  Total parameters: {model_info['total_params']:,}")


def create_experiment_directory(base_dir):
    """
    Create timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
    
    Returns:
        str: Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"FeatureModel_{timestamp}")
    
    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    
    print(f"✓ Created experiment directory: {experiment_dir}")
    
    return experiment_dir
