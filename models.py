import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Optional import for quantization features
try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False
    print("Warning: tensorflow_model_optimization not available. QAT features disabled.")


class ConfigurableCNN:
    """
    A configurable CNN model compatible with TFLite quantization.
    
    Features:
    - Accepts custom input and output dimensions
    - Simple CNN architecture with convolutional layers and one deep dense layer
    - TFLite compatible operations only
    - Quantization Aware Training (QAT) support
    """
    
    def __init__(self, input_shape, num_classes, dense_units=512):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input data (height, width, channels) or (length,) for 1D
            num_classes (int): Number of output classes/dimensions
            dense_units (int): Number of units in the deep dense layer (default: 512)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.model = None
        self.qat_model = None
        
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        # Determine if input is 1D or 2D based on input_shape
        if len(self.input_shape) == 1:
            # 1D CNN for time series or 1D data
            inputs = keras.Input(shape=self.input_shape)
            
            # Reshape for 1D convolution if needed
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
            
            # Convolutional layers
            x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        else:
            # 2D CNN for image-like data
            inputs = keras.Input(shape=self.input_shape)
            
            # Convolutional layers
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling2D()(x)
        
        # Deep dense layer
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dropout(0.3)(x)  # TFLite compatible dropout
        
        # Output layer
        if self.num_classes == 1:
            # Regression or binary classification
            outputs = layers.Dense(1, activation='linear')(x)
        elif self.num_classes == 2:
            # Binary classification with sigmoid
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001, loss=None, metrics=None):
        """
        Compile the model with appropriate loss function and metrics.
        
        Args:
            learning_rate (float): Learning rate for optimizer
            loss (str): Loss function (auto-detected if None)
            metrics (list): List of metrics (auto-detected if None)
        """
        if self.model is None:
            self.build_model()
        
        # Auto-detect loss function if not provided
        if loss is None:
            if self.num_classes == 1:
                loss = 'mse'  # Regression
            elif self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
        
        # Auto-detect metrics if not provided
        if metrics is None:
            if self.num_classes == 1:
                metrics = ['mae']  # Regression
            elif self.num_classes == 2:
                metrics = ['accuracy']
            else:
                metrics = ['accuracy']
        
        # Use Adam optimizer (TFLite compatible)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def get_model(self):
        """
        Get the compiled model.
        
        Returns:
            tf.keras.Model: The compiled model
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_for_tflite(self, filepath):
        """
        Save model in a format ready for TFLite conversion.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        # Save in SavedModel format (recommended for TFLite conversion)
        self.model.save(filepath, save_format='tf')
        print(f"Model saved to {filepath} (ready for TFLite conversion)")
    
    def prepare_qat_model(self, exclude_first_last=True):
        """
        Prepare model for Quantization Aware Training (QAT).
        
        Args:
            exclude_first_last (bool): Whether to exclude first/last layers from quantization
            
        Returns:
            tf.keras.Model: QAT-ready model
        """
        if not TFMOT_AVAILABLE:
            raise ImportError("tensorflow_model_optimization is required for QAT features")
            
        if self.model is None:
            self.build_model()
        
        # Define quantization config
        if exclude_first_last:
            # Custom quantization scheme that excludes first and last layers
            def apply_quantization_to_layer(layer):
                # Skip quantization for input layers and final dense layer
                if isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.Reshape)):
                    return layer
                if 'input' in layer.name.lower():
                    return layer
                # Skip final dense layer (output layer)
                if isinstance(layer, tf.keras.layers.Dense) and layer == self.model.layers[-1]:
                    return layer
                # Apply quantization to all other layers
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            
            # Apply quantization annotations
            annotated_model = tf.keras.utils.clone_model(
                self.model,
                clone_function=apply_quantization_to_layer,
            )
        else:
            # Full model quantization
            annotated_model = tfmot.quantization.keras.quantize_annotate_model(self.model)
        
        # Create QAT model
        self.qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        
        return self.qat_model
    
    def compile_qat_model(self, learning_rate=0.0001, loss=None, metrics=None):
        """
        Compile the QAT model with appropriate settings.
        
        Args:
            learning_rate (float): Learning rate for QAT (typically lower than FP training)
            loss (str): Loss function (auto-detected if None)
            metrics (list): List of metrics (auto-detected if None)
        """
        if self.qat_model is None:
            self.prepare_qat_model()
        
        # Auto-detect loss function if not provided
        if loss is None:
            if self.num_classes == 1:
                loss = 'mse'  # Regression
            elif self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
        
        # Auto-detect metrics if not provided
        if metrics is None:
            if self.num_classes == 1:
                metrics = ['mae']  # Regression
            elif self.num_classes == 2:
                metrics = ['accuracy']
            else:
                metrics = ['accuracy']
        
        # Use Adam optimizer with lower learning rate for QAT
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.qat_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def train_mixed_precision(self, x_train, y_train, x_val=None, y_val=None,
                             fp_epochs=5, qat_epochs=10, batch_size=32,
                             fp_lr=0.001, qat_lr=0.0001, callbacks=None,
                             exclude_first_last=True):
        """
        Train model with mixed precision: floating point first, then QAT.
        
        Args:
            x_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            x_val (np.ndarray): Validation data (optional)
            y_val (np.ndarray): Validation labels (optional)
            fp_epochs (int): Number of floating point training epochs
            qat_epochs (int): Number of QAT epochs
            batch_size (int): Batch size for training
            fp_lr (float): Learning rate for floating point training
            qat_lr (float): Learning rate for QAT training
            callbacks (list): List of Keras callbacks
            exclude_first_last (bool): Whether to exclude first/last layers from quantization
            
        Returns:
            dict: Training history for both phases
        """
        print("="*60)
        print("MIXED PRECISION TRAINING: FLOATING POINT + QAT")
        print("="*60)
        
        # Phase 1: Floating Point Training
        print(f"\nPhase 1: Floating Point Training ({fp_epochs} epochs)")
        print("-" * 50)
        
        if self.model is None:
            self.build_model()
            self.compile_model(learning_rate=fp_lr)
        
        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
        
        fp_history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=fp_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Floating point training completed. Final loss: {fp_history.history['loss'][-1]:.4f}")
        
        # Phase 2: QAT Training
        print(f"\nPhase 2: Quantization Aware Training ({qat_epochs} epochs)")
        print("-" * 50)
        
        # Prepare QAT model using the trained weights
        self.prepare_qat_model(exclude_first_last=exclude_first_last)
        self.compile_qat_model(learning_rate=qat_lr)
        
        # Copy weights from FP model to QAT model
        self.qat_model.set_weights(self.model.get_weights())
        
        qat_history = self.qat_model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=qat_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"QAT training completed. Final loss: {qat_history.history['loss'][-1]:.4f}")
        
        # Combine histories
        combined_history = {
            'fp_history': fp_history.history,
            'qat_history': qat_history.history,
            'total_epochs': fp_epochs + qat_epochs
        }
        
        print("\n" + "="*60)
        print("MIXED PRECISION TRAINING COMPLETED")
        print("="*60)
        
        return combined_history
    
    def get_qat_model(self):
        """
        Get the QAT model.
        
        Returns:
            tf.keras.Model: The QAT model
        """
        if self.qat_model is None:
            self.prepare_qat_model()
            self.compile_qat_model()
        return self.qat_model
    
    def save_qat_model(self, filepath):
        """
        Save QAT model for TFLite conversion.
        
        Args:
            filepath (str): Path to save the QAT model
        """
        if self.qat_model is None:
            raise ValueError("QAT model must be prepared before saving")
        
        # Save in SavedModel format
        self.qat_model.save(filepath, save_format='tf')
        print(f"QAT model saved to {filepath} (ready for TFLite conversion)")


# Example usage functions
def create_image_cnn(height, width, channels, num_classes):
    """
    Create a CNN for image classification.
    
    Args:
        height (int): Image height
        width (int): Image width  
        channels (int): Number of channels (1 for grayscale, 3 for RGB)
        num_classes (int): Number of output classes
    
    Returns:
        ConfigurableCNN: Configured CNN model
    """
    input_shape = (height, width, channels)
    cnn = ConfigurableCNN(input_shape, num_classes)
    return cnn


def create_timeseries_cnn(sequence_length, num_classes):
    """
    Create a CNN for time series or 1D signal classification.
    
    Args:
        sequence_length (int): Length of input sequence
        num_classes (int): Number of output classes
    
    Returns:
        ConfigurableCNN: Configured CNN model
    """
    input_shape = (sequence_length,)
    cnn = ConfigurableCNN(input_shape, num_classes)
    return cnn


# Example usage
if __name__ == "__main__":
    # Example 1: Image classification (28x28 grayscale, 10 classes)
    print("Example 1: Image CNN")
    image_cnn = create_image_cnn(28, 28, 1, 10)
    model1 = image_cnn.get_model()
    print(image_cnn.summary())
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Time series classification (100 timesteps, 5 classes)
    print("Example 2: Time Series CNN")
    ts_cnn = create_timeseries_cnn(100, 5)
    model2 = ts_cnn.get_model()
    print(ts_cnn.summary())
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Custom configuration
    print("Example 3: Custom CNN")
    custom_cnn = ConfigurableCNN(input_shape=(64, 64, 3), num_classes=1, dense_units=256)
    model3 = custom_cnn.get_model()
    print(custom_cnn.summary())
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: QAT Training Demo
    print("Example 4: QAT Training Demo")
    print("Note: This is a demonstration of the QAT training interface.")
    print("For actual training, provide real data to train_mixed_precision().")
    
    qat_cnn = create_image_cnn(28, 28, 1, 10)
    
    # Simulate training data shapes (replace with real data)
    print("QAT model prepared. Use the following pattern for training:")
    print("""
    # Example training call:
    history = qat_cnn.train_mixed_precision(
        x_train=your_training_data,
        y_train=your_training_labels,
        x_val=your_validation_data,
        y_val=your_validation_labels,
        fp_epochs=5,      # Floating point epochs
        qat_epochs=10,    # QAT epochs
        batch_size=32,
        fp_lr=0.001,      # FP learning rate
        qat_lr=0.0001,    # QAT learning rate (lower)
        exclude_first_last=True
    )
    
    # Save QAT model for TFLite conversion
    qat_cnn.save_qat_model('./qat_model')
    """)


def create_deep_efficient_cnn(input_shape=(2, 7, 256), num_classes=10, embedding_size=64):
    """
    Create a deep CNN model following MobileNet-inspired inverted bottleneck design
    Pattern: narrow → wide → narrow for memory efficiency
    Target: <200KB total memory consumption
    """
    print(f"Creating deep CNN model with input_shape={input_shape}, num_classes={num_classes}")
    print(f"Target embedding size: {embedding_size}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # === EXPANSION PHASE: narrow → wide ===
        # Block 1: Start narrow (preserve height=7, reduce width gradually)
        tf.keras.layers.Conv2D(16, (1, 3), activation='relu', padding='same', name='conv1_expand'),
        tf.keras.layers.BatchNormalization(name='bn1'),
        
        # Block 2: Expand channels
        tf.keras.layers.Conv2D(32, (1, 3), activation='relu', padding='same', name='conv2_expand'),
        tf.keras.layers.BatchNormalization(name='bn2'),
        
        # Block 3: Continue expansion
        tf.keras.layers.Conv2D(48, (1, 3), activation='relu', padding='same', name='conv3_expand'),
        tf.keras.layers.BatchNormalization(name='bn3'),
        
        # === MIDDLE PHASE: Peak width processing ===
        # Block 4: Peak channels - no pooling to preserve spatial dimensions
        tf.keras.layers.Conv2D(64, (1, 3), activation='relu', padding='same', name='conv4_peak'),
        tf.keras.layers.BatchNormalization(name='bn4'),
        
        # Block 5: Maintain peak with smaller kernel
        tf.keras.layers.Conv2D(64, (1, 2), activation='relu', padding='same', name='conv5_peak'),
        tf.keras.layers.BatchNormalization(name='bn5'),
        
        # === COMPRESSION PHASE: wide → narrow ===
        # Block 6: Start compression - no pooling to avoid dimension collapse
        tf.keras.layers.Conv2D(48, (1, 3), activation='relu', padding='same', name='conv6_compress'),
        tf.keras.layers.BatchNormalization(name='bn6'),
        
        # Block 7: Further compression - no pooling to avoid dimension issues
        tf.keras.layers.Conv2D(32, (1, 2), activation='relu', padding='same', name='conv7_compress'),
        tf.keras.layers.BatchNormalization(name='bn7'),
        
        # Block 8: Final compression to target embedding
        tf.keras.layers.Conv2D(embedding_size, (1, 2), activation='relu', padding='same', name='conv8_final'),
        tf.keras.layers.BatchNormalization(name='bn8'),
        
        # === GLOBAL POOLING & CLASSIFICATION ===
        # Preserve spatial information while creating 1D embedding
        tf.keras.layers.GlobalAveragePooling2D(name='global_pool'),  # → (embedding_size,)
        
        # Small dense layer for final classification
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use categorical_crossentropy for one-hot encoded labels
        metrics=['accuracy']
    )
    
    # Set model name for experiment tracking
    model._name = 'DeepEfficientCNN'
    
    print("Deep efficient CNN model created and compiled successfully!")
    
    return model


def create_ultra_lightweight_cnn(input_shape=(2, 7, 256), num_classes=3, embedding_size=32):
    """
    Create an ultra-lightweight CNN model optimized for <150KB memory consumption.
    Removes batch normalization and uses input normalization instead.
    Reduces layers and channels to minimize parameters.
    
    Args:
        input_shape: Input tensor shape (channels, height, width)
        num_classes: Number of output classes
        embedding_size: Final embedding dimension (reduced from 64 to 32)
    """
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
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set model name for experiment tracking
    model._name = 'UltraLightweightCNN'
    
    print("Ultra-lightweight CNN model created and compiled successfully!")
    
    return model


def create_simple_cnn(input_shape, num_classes):
    """
    Create a CNN model appropriate for small seismic data (2, 7, 256)
    """
    print(f"Creating CNN model with input_shape={input_shape}, num_classes={num_classes}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First conv block - use smaller kernels and stride instead of pooling
        tf.keras.layers.Conv2D(32, (2, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Second conv block - use (1,2) pooling to preserve spatial dimensions
        tf.keras.layers.Conv2D(64, (2, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((1, 2)),  # Only pool width, preserve height
        tf.keras.layers.BatchNormalization(),
        
        # Third conv block
        tf.keras.layers.Conv2D(128, (1, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((1, 2)),  # Only pool width again
        tf.keras.layers.BatchNormalization(),
        
        # Global pooling instead of flattening
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set model name for experiment tracking
    model._name = 'SimpleCNN'
    
    print("Model created successfully!")
    print("\nModel Summary:")
    model.summary()
    
    return model


def print_model_info(model):
    """Print model architecture and memory usage"""
    model.summary()
    
    # Calculate memory usage
    total_params = model.count_params()
    # Assuming float32 (4 bytes per parameter)
    memory_kb = (total_params * 4) / 1024
    
    print(f"\n=== MEMORY ANALYSIS ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated parameter memory: {memory_kb:.2f} KB")
    print(f"Target: <200 KB - {'✅ PASS' if memory_kb < 200 else '❌ EXCEED'}")
    
    # Test forward pass to verify output size
    test_input = tf.random.normal((1, 2, 7, 256))
    
    # Build the model by doing a forward pass first
    output = model(test_input)
    
    # === Activation memory estimation ===
    # Assumption: sequential execution, at any time we keep current layer input and output,
    # then immediately free the input after output is computed.
    try:
        # Resolve dtype size in bytes (fallback to 4 for float32)
        model_dtype = getattr(model, 'dtype', None) or tf.float32
        bytes_per_element = getattr(tf.dtypes.as_dtype(model_dtype), 'size', 4) or 4
        
        # Collect per-layer inputs and outputs as tensors
        all_layer_inputs = []
        all_layer_outputs = []
        layer_io_slices = []  # (start_idx_inputs, len_inputs, start_idx_outputs, len_outputs)
        
        def flatten_tensors(obj):
            flat = []
            for t in tf.nest.flatten(obj):
                if isinstance(t, (tf.Tensor, tf.Variable)) or hasattr(t, 'dtype'):
                    flat.append(t)
            return flat
        
        for layer in model.layers:
            layer_inputs = flatten_tensors(getattr(layer, 'input', []))
            layer_outputs = flatten_tensors(getattr(layer, 'output', []))
            i_start = len(all_layer_inputs)
            o_start = len(all_layer_outputs)
            all_layer_inputs.extend(layer_inputs)
            all_layer_outputs.extend(layer_outputs)
            layer_io_slices.append((i_start, len(layer_inputs), o_start, len(layer_outputs), layer.name))
        
        # Build a probing model to fetch concrete tensors for shapes after forward pass
        probe_outputs = all_layer_inputs + all_layer_outputs
        activation_peak_kb = 0.0
        if len(probe_outputs) > 0:
            probe_model = tf.keras.Model(inputs=model.inputs, outputs=probe_outputs)
            concrete = probe_model(test_input, training=False)
            # Ensure list
            if not isinstance(concrete, (list, tuple)):
                concrete = [concrete]
            
            # Split back to inputs/outputs
            num_inputs_total = len(all_layer_inputs)
            concrete_inputs = concrete[:num_inputs_total]
            concrete_outputs = concrete[num_inputs_total:]
            
            # Compute peak over layers: sum(size(inputs)) + sum(size(outputs)) per layer
            for (i_start, i_len, o_start, o_len, lname) in layer_io_slices:
                # sum elements for inputs
                in_elems = 0
                for t in concrete_inputs[i_start:i_start + i_len]:
                    try:
                        in_elems += int(tf.size(t))
                    except Exception:
                        shape = t.shape
                        if None in shape:
                            continue
                        prod = 1
                        for d in shape:
                            prod *= int(d)
                        in_elems += prod
                # sum elements for outputs
                out_elems = 0
                for t in concrete_outputs[o_start:o_start + o_len]:
                    try:
                        out_elems += int(tf.size(t))
                    except Exception:
                        shape = t.shape
                        if None in shape:
                            continue
                        prod = 1
                        for d in shape:
                            prod *= int(d)
                        out_elems += prod
                layer_bytes = (in_elems + out_elems) * bytes_per_element
                layer_kb = layer_bytes / 1024.0
                if layer_kb > activation_peak_kb:
                    activation_peak_kb = layer_kb
        
        print(f"Estimated peak activation memory (input+output of one layer): {activation_peak_kb:.2f} KB")
        print(f"Estimated peak total (params + activations): {memory_kb + activation_peak_kb:.2f} KB")
    except Exception as e:
        print(f"Warning: Activation memory estimation failed: {e}")
    
    print(f"\n=== OUTPUT VERIFICATION ===")
    print(f"Input shape: {test_input.shape}")
    print(f"Final output shape: {output.shape}")
    print(f"Target embedding size: 64 (from global_pool layer)")
    
    return memory_kb


def create_resnet_style_cnn(input_shape=(2, 7, 256), num_classes=10, target_memory_kb=3000):
    """
    Create a ResNet-style CNN with residual connections optimized for 3MB memory target.
    Designed for seismic sensor data with input shape (2, 7, 256).
    Enhanced version with deeper architecture and wider channels for improved accuracy.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        target_memory_kb: Target memory consumption in KB
    
    Returns:
        tf.keras.Model: ResNet-style CNN model
    """
    print(f"Creating Enhanced ResNet-style CNN with input_shape={input_shape}, num_classes={num_classes}")
    print(f"Target memory: {target_memory_kb} KB")
    
    def efficient_residual_block(x, filters, kernel_size=(1, 3), stride=(1, 1), name_prefix="res"):
        """
        Efficient residual block with skip connection - reduced parameters
        """
        shortcut = x
        
        # First conv layer with smaller filters
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=stride, padding='same', 
            name=f'{name_prefix}_conv1'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu1')(x)
        
        # Second conv layer
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', 
            name=f'{name_prefix}_conv2'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Adjust shortcut if needed (channel or spatial dimension mismatch)
        if shortcut.shape[-1] != filters or stride != (1, 1):
            shortcut = tf.keras.layers.Conv2D(
                filters, (1, 1), strides=stride, padding='same',
                name=f'{name_prefix}_shortcut_conv'
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(
                name=f'{name_prefix}_shortcut_bn'
            )(shortcut)
        
        # Add skip connection
        x = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu2')(x)
        
        return x
    
    def lightweight_bottleneck_block(x, filters, kernel_size=(1, 3), stride=(1, 1), name_prefix="bottleneck"):
        """
        Lightweight bottleneck residual block - more aggressive compression
        """
        shortcut = x
        
        # 1x1 conv to reduce dimensions (optimized for 3MB budget)
        reduced_filters = max(filters // 4, 16)  # Less aggressive reduction for better capacity
        x = tf.keras.layers.Conv2D(
            reduced_filters, (1, 1), strides=stride, padding='same',
            name=f'{name_prefix}_conv1'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu1')(x)
        
        # 3x3 conv (depthwise separable for efficiency)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size, padding='same',
            name=f'{name_prefix}_depthwise'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu2')(x)
        
        # 1x1 conv to restore dimensions
        x = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same',
            name=f'{name_prefix}_conv3'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters or stride != (1, 1):
            shortcut = tf.keras.layers.Conv2D(
                filters, (1, 1), strides=stride, padding='same',
                name=f'{name_prefix}_shortcut_conv'
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(
                name=f'{name_prefix}_shortcut_bn'
            )(shortcut)
        
        # Add skip connection
        x = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu3')(x)
        
        return x
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    
    # Initial conv layer (stem) - optimized for 3MB budget
    x = tf.keras.layers.Conv2D(48, (1, 7), strides=(1, 2), padding='same', name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)
    # Shape: (2, 7, 128)
    
    # Stage 1: Enhanced residual blocks with balanced capacity
    x = efficient_residual_block(x, 64, name_prefix='stage1_block1')
    x = efficient_residual_block(x, 64, name_prefix='stage1_block2')
    x = efficient_residual_block(x, 64, name_prefix='stage1_block3')
    # Shape: (2, 7, 128)
    
    # Stage 2: Increase channels moderately, reduce width
    x = efficient_residual_block(x, 96, stride=(1, 2), name_prefix='stage2_block1')  # Downsample width
    x = efficient_residual_block(x, 96, name_prefix='stage2_block2')
    x = efficient_residual_block(x, 96, name_prefix='stage2_block3')
    x = efficient_residual_block(x, 96, name_prefix='stage2_block4')
    # Shape: (2, 7, 64)
    
    # Stage 3: Bottleneck blocks with controlled capacity
    x = lightweight_bottleneck_block(x, 128, stride=(1, 2), name_prefix='stage3_block1')  # Downsample width
    x = lightweight_bottleneck_block(x, 128, name_prefix='stage3_block2')
    x = lightweight_bottleneck_block(x, 128, name_prefix='stage3_block3')
    x = lightweight_bottleneck_block(x, 128, name_prefix='stage3_block4')
    x = lightweight_bottleneck_block(x, 128, name_prefix='stage3_block5')
    # Shape: (2, 7, 32)
    
    # Stage 4: Higher capacity feature extraction
    x = lightweight_bottleneck_block(x, 160, name_prefix='stage4_block1')
    x = lightweight_bottleneck_block(x, 160, name_prefix='stage4_block2')
    x = lightweight_bottleneck_block(x, 160, name_prefix='stage4_block3')
    # Shape: (2, 7, 32)
    
    # Stage 5: Final feature extraction (additional depth for 3MB budget)
    x = lightweight_bottleneck_block(x, 192, name_prefix='stage5_block1')
    x = lightweight_bottleneck_block(x, 192, name_prefix='stage5_block2')
    # Shape: (2, 7, 32)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    # Shape: (192,)
    
    # Enhanced dense layers for better feature processing (scaled appropriately)
    x = tf.keras.layers.Dense(384, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout1')(x)
    x = tf.keras.layers.Dense(192, activation='relu', name='dense2')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout2')(x)
    
    # Final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    
    # Create model
    model = tf.keras.Model(inputs, outputs, name='EnhancedResNet_SeismicCNN_3MB')
    
    # Set model name for experiment tracking
    model._name = 'ResNetStyleCNN'
    
    return model


def deep_cnn_large(input_shape=(2, 7, 256), num_classes=10):
    """
    Create a large, deep CNN architecture for seismic sensor data.
    Memory is not constrained, so this uses a substantial architecture with many layers.
    No MaxPooling layers are used to preserve spatial dimensions throughout the network.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        tf.keras.Model: Large deep CNN model without pooling layers
    """
    print(f"Creating Deep CNN Large with input_shape={input_shape}, num_classes={num_classes}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input'),
        
        # === CONVOLUTIONAL BLOCK 1 ===
        tf.keras.layers.Conv2D(64, (2, 3), activation='relu', padding='same', name='conv1_1'),
        tf.keras.layers.BatchNormalization(name='bn1_1'),
        tf.keras.layers.Conv2D(64, (2, 3), activation='relu', padding='same', name='conv1_2'),
        tf.keras.layers.BatchNormalization(name='bn1_2'),
        
        # === CONVOLUTIONAL BLOCK 2 ===
        tf.keras.layers.Conv2D(128, (1, 3), activation='relu', padding='same', name='conv2_1'),
        tf.keras.layers.BatchNormalization(name='bn2_1'),
        tf.keras.layers.Conv2D(128, (1, 3), activation='relu', padding='same', name='conv2_2'),
        tf.keras.layers.BatchNormalization(name='bn2_2'),
        tf.keras.layers.Conv2D(128, (1, 3), activation='relu', padding='same', name='conv2_3'),
        tf.keras.layers.BatchNormalization(name='bn2_3'),
        
        # === CONVOLUTIONAL BLOCK 3 ===
        tf.keras.layers.Conv2D(256, (1, 3), activation='relu', padding='same', name='conv3_1'),
        tf.keras.layers.BatchNormalization(name='bn3_1'),
        tf.keras.layers.Conv2D(256, (1, 3), activation='relu', padding='same', name='conv3_2'),
        tf.keras.layers.BatchNormalization(name='bn3_2'),
        tf.keras.layers.Conv2D(256, (1, 3), activation='relu', padding='same', name='conv3_3'),
        tf.keras.layers.BatchNormalization(name='bn3_3'),
        
        # === CONVOLUTIONAL BLOCK 4 ===
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv4_1'),
        tf.keras.layers.BatchNormalization(name='bn4_1'),
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv4_2'),
        tf.keras.layers.BatchNormalization(name='bn4_2'),
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv4_3'),
        tf.keras.layers.BatchNormalization(name='bn4_3'),
        
        # === CONVOLUTIONAL BLOCK 5 ===
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv5_1'),
        tf.keras.layers.BatchNormalization(name='bn5_1'),
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv5_2'),
        tf.keras.layers.BatchNormalization(name='bn5_2'),
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv5_3'),
        tf.keras.layers.BatchNormalization(name='bn5_3'),
        
        # === CONVOLUTIONAL BLOCK 6 ===
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv6_1'),
        tf.keras.layers.BatchNormalization(name='bn6_1'),
        tf.keras.layers.Conv2D(512, (1, 3), activation='relu', padding='same', name='conv6_2'),
        tf.keras.layers.BatchNormalization(name='bn6_2'),
        
        # === GLOBAL POOLING ===
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # === DENSE LAYERS ===
        tf.keras.layers.Dense(2048, activation='relu', name='dense1'),
        tf.keras.layers.Dropout(0.5, name='dropout1'),
        tf.keras.layers.Dense(1024, activation='relu', name='dense2'),
        tf.keras.layers.Dropout(0.5, name='dropout2'),
        tf.keras.layers.Dense(512, activation='relu', name='dense3'),
        tf.keras.layers.Dropout(0.3, name='dropout3'),
        tf.keras.layers.Dense(256, activation='relu', name='dense4'),
        tf.keras.layers.Dropout(0.2, name='dropout4'),
        
        # === OUTPUT LAYER ===
        tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set model name for experiment tracking
    model._name = 'DeepCNNLarge'
    
    print("Deep CNN Large model created successfully!")
    print("\nModel Summary:")
    model.summary()
    
    return model


def print_resnet_model_info(model, target_memory_kb=3000):
    """Print ResNet model architecture and memory usage with 3MB target"""
    model.summary()
    
    # Calculate memory usage
    total_params = model.count_params()
    # Assuming float32 (4 bytes per parameter)
    memory_kb = (total_params * 4) / 1024
    
    print(f"\n=== RESNET MODEL MEMORY ANALYSIS ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory: {memory_kb:.2f} KB")
    print(f"Target: <{target_memory_kb} KB - {'✅ PASS' if memory_kb < target_memory_kb else '❌ EXCEED'}")
    
    if memory_kb > target_memory_kb:
        print(f"⚠️  Model exceeds target by {memory_kb - target_memory_kb:.2f} KB")
        print("💡 Consider reducing filters or using more bottleneck blocks")
    else:
        print(f"✅ Model is {target_memory_kb - memory_kb:.2f} KB under target")
    
    # Test forward pass to verify output size
    test_input = tf.random.normal((1, 2, 7, 256))
    
    # Build the model by doing a forward pass first
    output = model(test_input)
    
    print(f"\n=== OUTPUT VERIFICATION ===")
    print(f"Input shape: {test_input.shape}")
    print(f"Final output shape: {output.shape}")
    print(f"Expected output shape: (1, {model.layers[-1].units})")
    
    return memory_kb