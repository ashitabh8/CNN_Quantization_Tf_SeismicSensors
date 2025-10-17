import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def create_seismic_cnn_model(config,input_shape,embedding_size=32):
    """
    Create a simple CNN model for seismic sensor data classification.
    
    Args:
        input_shape: Input shape tuple (e.g., (10, 11) for Welch spectrogram or (200,) for raw data)
        num_classes: Number of output classes
        model_name: Name for the model
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    num_classes = len(config['vehicle_classification']['included_classes'])
    if num_classes == 0:
        num_classes = len(config['vehicle_classification']['class_names'])
    model_name = "SeismicCNN"

    print(f"Creating ultra-lightweight CNN model with input_shape={input_shape}, num_classes={num_classes}")
    print(f"Target embedding size: {embedding_size}")
    print("Using input normalization instead of batch normalization")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # === INPUT NORMALIZATION ===
        # Per-input normalization to replace batch normalization
        # tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=[1, 2, 3]), name='input_norm'),
        
        # === COMPRESSED ARCHITECTURE ===
        # Block 1: Start with fewer channels
        tf.keras.layers.Conv2D(12, (1, 3), activation='relu', padding='same', name='conv1'),
        
        # Block 2: Moderate expansion
        tf.keras.layers.Conv2D(24, (1, 3), activation='relu', padding='same', name='conv2'),
        
        # Block 3: Peak processing with reduced channels
        tf.keras.layers.Conv2D(32, (1, 3), activation='relu', padding='same', name='conv3'),
        
        # Block 4: Compression phase
        tf.keras.layers.Conv2D(24, (1, 2), activation='relu', padding='same', name='conv4'),
        
        # Block 5: Final compression to embedding
        tf.keras.layers.Conv2D(embedding_size, (1, 2), activation='relu', padding='same', name='conv5_final'),
        
        # === GLOBAL POOLING & CLASSIFICATION ===
        tf.keras.layers.GlobalAveragePooling2D(name='global_pool'),
        
        # Direct classification without additional dense layers
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ], name=model_name)
    
    
    return model


def create_simple_1d_cnn_model(input_shape, num_classes, model_name="Seismic1DCNN"):
    """
    Create a simple 1D CNN model for seismic sensor data classification.
    This is more suitable for raw time series data.
    
    Args:
        input_shape: Input shape tuple (e.g., (200,) for raw data)
        num_classes: Number of output classes
        model_name: Name for the model
    
    Returns:
        tf.keras.Model: Compiled 1D CNN model
    """
    model = tf.keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input'),
        
        # Reshape for 1D convolution
        layers.Reshape((*input_shape, 1), name='reshape_to_1d'),
        
        # First 1D convolutional block
        layers.Conv1D(32, 3, activation='relu', padding='same', name='conv1d_1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling1D(2, name='pool1d_1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second 1D convolutional block
        layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling1D(2, name='pool1d_2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third 1D convolutional block
        layers.Conv1D(128, 3, activation='relu', padding='same', name='conv1d_3'),
        layers.BatchNormalization(name='bn3'),
        layers.GlobalAveragePooling1D(name='global_avg_pool'),
        layers.Dropout(0.5, name='dropout3'),
        
        # Dense layer
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout4'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name=model_name)
    
    return model


def compile_model(config, model):
    """
    Compile the model with appropriate optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=config['loss'],
        metrics=config['metrics']
    )
    
    return model


def get_model_summary(model):
    """
    Get a detailed summary of the model architecture.
    
    Args:
        model: Keras model
    
    Returns:
        str: Model summary
    """
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    summary = buffer.getvalue()
    sys.stdout = old_stdout
    
    return summary


def count_parameters(model):
    """
    Count the number of trainable and non-trainable parameters in the model.
    
    Args:
        model: Keras model
    
    Returns:
        dict: Parameter counts
    """
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    return {
        'trainable': int(trainable_params),
        'non_trainable': int(non_trainable_params),
        'total': int(trainable_params + non_trainable_params)
    }
