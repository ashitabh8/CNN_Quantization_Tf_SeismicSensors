import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def create_seismic_cnn_model(config,input_shape):
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
    model = tf.keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input'),
        
        # Reshape for 2D convolution if input is 1D
        layers.Reshape((*input_shape, 1), name='reshape_to_2d') if len(input_shape) == 1 else layers.Lambda(lambda x: x, name='keep_2d'),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.Dropout(0.5, name='dropout3'),
        
        # Dense layer
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout4'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name=model_name)
    
    # Set model name for logging
    model._name = model_name
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=config['loss'],
        metrics=config['metrics']
    )
    
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


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
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
