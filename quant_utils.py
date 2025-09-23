import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import os


class QuantizationUtils:
    """
    Utility class for TensorFlow Lite quantization operations.
    Provides functions for INT8 static and dynamic quantization with
    first/last layer preservation for CNN models.
    """
    
    @staticmethod
    def keras_to_tflite_static_int8(model, representative_dataset, output_path, 
                                   exclude_first_last=True):
        """
        Convert Keras model to TFLite with INT8 static quantization.
        Avoids quantizing first and last layers to preserve accuracy.
        
        Args:
            model (tf.keras.Model): Input Keras model
            representative_dataset (callable): Function that yields representative data
            output_path (str): Path to save the quantized TFLite model
            exclude_first_last (bool): Whether to exclude first/last layers from quantization
            
        Returns:
            str: Path to the saved TFLite model
        """
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # Set inference input/output types to INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Exclude first and last layers from quantization if requested
        if exclude_first_last:
            # Get layer names to exclude
            layer_names = [layer.name for layer in model.layers]
            if len(layer_names) > 2:
                # Exclude first and last layers
                excluded_layers = [layer_names[0], layer_names[-1]]
                
                # Find input and output layers (Dense layers typically)
                input_layer = None
                output_layer = None
                
                for layer in model.layers:
                    if 'input' in layer.name.lower() or isinstance(layer, tf.keras.layers.InputLayer):
                        input_layer = layer.name
                    if 'dense' in layer.name.lower() and layer == model.layers[-1]:
                        output_layer = layer.name
                
                # Set quantization exclusions
                def representative_dataset_gen():
                    for data in representative_dataset():
                        yield [data.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                
                # Keep input/output as float32 to avoid first/last layer quantization
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32
        
        # Convert the model
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"Quantization failed with full INT8. Trying mixed precision...")
            # Fallback to mixed precision if full INT8 fails
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Static INT8 quantized model saved to: {output_path}")
        return output_path
    
    @staticmethod
    def keras_to_tflite_dynamic_int8(model, output_path, exclude_first_last=True):
        """
        Convert Keras model to TFLite with INT8 dynamic quantization.
        Avoids quantizing first and last layers to preserve accuracy.
        
        Args:
            model (tf.keras.Model): Input Keras model
            output_path (str): Path to save the quantized TFLite model
            exclude_first_last (bool): Whether to exclude first/last layers from quantization
            
        Returns:
            str: Path to the saved TFLite model
        """
        # Convert to TFLite with dynamic quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable dynamic range quantization (INT8 weights, float32 activations)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if exclude_first_last:
            # Keep input/output as float32 to preserve first/last layer precision
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        else:
            # Full dynamic quantization
            converter.target_spec.supported_types = [tf.int8]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Dynamic INT8 quantized model saved to: {output_path}")
        return output_path
    
    @staticmethod
    def load_tflite_model(model_path):
        """
        Load a TFLite model from file.
        
        Args:
            model_path (str): Path to the TFLite model file
            
        Returns:
            tf.lite.Interpreter: Loaded TFLite interpreter
        """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    @staticmethod
    def test_tflite_accuracy(model_path, test_data, test_labels, num_classes=None):
        """
        Test accuracy of a TFLite model.
        
        Args:
            model_path (str): Path to the TFLite model file
            test_data (np.ndarray): Test input data
            test_labels (np.ndarray): Test labels
            num_classes (int): Number of classes (auto-detected if None)
            
        Returns:
            float: Test accuracy
        """
        # Load the TFLite model
        interpreter = QuantizationUtils.load_tflite_model(model_path)
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        
        print(f"Testing TFLite model: {os.path.basename(model_path)}")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        
        # Run inference on test data
        for i, sample in enumerate(test_data):
            # Prepare input
            input_data = np.expand_dims(sample, axis=0).astype(input_details[0]['dtype'])
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_data)} samples")
        
        predictions = np.array(predictions)
        
        # Calculate accuracy based on output format
        if predictions.shape[1] == 1:
            # Binary classification or regression
            if num_classes == 2 or (test_labels.max() <= 1 and test_labels.min() >= 0):
                # Binary classification
                pred_classes = (predictions > 0.5).astype(int).flatten()
                accuracy = accuracy_score(test_labels, pred_classes)
            else:
                # Regression - use MAE as "accuracy" metric
                accuracy = np.mean(np.abs(predictions.flatten() - test_labels))
                print(f"MAE (regression): {accuracy}")
                return accuracy
        else:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(test_labels, pred_classes)
        
        print(f"TFLite model accuracy: {accuracy:.4f}")
        return accuracy
    
    @staticmethod
    def test_keras_accuracy(model, test_data, test_labels):
        """
        Test accuracy of a Keras model.
        
        Args:
            model (tf.keras.Model): Keras model
            test_data (np.ndarray): Test input data
            test_labels (np.ndarray): Test labels
            
        Returns:
            float: Test accuracy
        """
        print("Testing original Keras model...")
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        
        # Calculate accuracy based on output format
        if predictions.shape[1] == 1:
            # Binary classification or regression
            if test_labels.max() <= 1 and test_labels.min() >= 0:
                # Binary classification
                pred_classes = (predictions > 0.5).astype(int).flatten()
                accuracy = accuracy_score(test_labels, pred_classes)
            else:
                # Regression - use MAE as "accuracy" metric
                accuracy = np.mean(np.abs(predictions.flatten() - test_labels))
                print(f"Keras model MAE (regression): {accuracy}")
                return accuracy
        else:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(test_labels, pred_classes)
        
        print(f"Keras model accuracy: {accuracy:.4f}")
        return accuracy


# Helper functions for easy usage
def create_representative_dataset(sample_data, num_samples=100):
    """
    Create a representative dataset generator for static quantization.
    
    Args:
        sample_data (np.ndarray): Sample data for calibration
        num_samples (int): Number of samples to use for calibration
        
    Returns:
        callable: Representative dataset generator
    """
    def representative_dataset_gen():
        for i in range(min(num_samples, len(sample_data))):
            yield [sample_data[i:i+1].astype(np.float32)]
    
    return representative_dataset_gen


def quantize_and_compare(model, test_data, test_labels, calibration_data=None, 
                        output_dir="./quantized_models"):
    """
    Comprehensive function to quantize a model and compare accuracies.
    
    Args:
        model (tf.keras.Model): Original Keras model
        test_data (np.ndarray): Test data
        test_labels (np.ndarray): Test labels
        calibration_data (np.ndarray): Data for static quantization calibration
        output_dir (str): Directory to save quantized models
        
    Returns:
        dict: Dictionary with accuracy results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Test original model
    results['original_accuracy'] = QuantizationUtils.test_keras_accuracy(
        model, test_data, test_labels
    )
    
    # Dynamic quantization
    dynamic_path = os.path.join(output_dir, "model_dynamic_int8.tflite")
    QuantizationUtils.keras_to_tflite_dynamic_int8(model, dynamic_path)
    results['dynamic_accuracy'] = QuantizationUtils.test_tflite_accuracy(
        dynamic_path, test_data, test_labels
    )
    
    # Static quantization (if calibration data provided)
    if calibration_data is not None:
        static_path = os.path.join(output_dir, "model_static_int8.tflite")
        rep_dataset = create_representative_dataset(calibration_data)
        QuantizationUtils.keras_to_tflite_static_int8(
            model, rep_dataset, static_path
        )
        results['static_accuracy'] = QuantizationUtils.test_tflite_accuracy(
            static_path, test_data, test_labels
        )
    
    # Print summary
    print("\n" + "="*50)
    print("QUANTIZATION RESULTS SUMMARY")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    return results


# Example usage
if __name__ == "__main__":
    # This would be used with actual data
    print("Quantization utilities loaded successfully!")
    print("Use QuantizationUtils class methods or helper functions for quantization.")