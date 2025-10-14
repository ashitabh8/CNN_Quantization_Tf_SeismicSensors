import tensorflow as tf
import numpy as np
import os
import yaml
from datetime import datetime
import json
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow_model_optimization as tfmot

from models import create_seismic_cnn_model, create_simple_1d_cnn_model, compile_model, count_parameters
from tensorboard_utils import TensorBoardLogger, create_experiment_id, setup_experiment_logging
from dataset import SeismicDataset, Args
from dataset_utils import create_mapping_vehicle_name_to_file_path, filter_samples_by_max_distance


class SeismicModelTrainer:
    """
    Complete training and evaluation pipeline for seismic sensor models.
    """
    
    def __init__(self, config_path, model_name="SeismicCNN"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to dataset configuration file
            model_name: Name of the model
        """
        self.config_path = config_path
        self.model_name = model_name
        self.config = None
        self.model = None
        self.class_names = None
        self.experiment_id = None
        self.log_dirs = None
        self.tensorboard_logger = None
        
        # Load configuration
        self._load_config()
        
        # Setup experiment
        self._setup_experiment()
    
    def _load_config(self):
        """Load dataset configuration."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract class names
        if len(self.config['vehicle_classification']['included_classes']) == 0:
            self.class_names = self.config['vehicle_classification']['class_names']
        else:
            self.class_names = self.config['vehicle_classification']['included_classes']
        print(f"Loaded configuration with classes: {self.class_names}")
    
    def _setup_experiment(self):
        """Setup experiment logging and directories."""
        self.experiment_id = create_experiment_id(self.model_name)
        self.log_dirs = setup_experiment_logging(self.experiment_id)
        
        # Initialize TensorBoard logger
        self.tensorboard_logger = TensorBoardLogger(
            self.log_dirs['tensorboard_dir'], 
            self.class_names
        )
        
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Log directories: {self.log_dirs}")
    
    def _setup_datasets(self):
        """Setup training and validation datasets."""
        # Create dataset mappings
        train_mapping, val_mapping = create_mapping_vehicle_name_to_file_path(self.config)
        train_mapping = filter_samples_by_max_distance(train_mapping, self.config['max_distance_m'])
        val_mapping = filter_samples_by_max_distance(val_mapping, self.config['max_distance_m'])
        
        batch_size = self.config['batch_size']
        
        # Create datasets
        train_dataset = SeismicDataset(
            train_mapping=train_mapping,
            val_mapping=val_mapping,
            task="vehicle_classification",
            spectral_processing=self.config.get('spectral_processing', {}),
            is_training=True
        )
        
        val_dataset = SeismicDataset(
            train_mapping=train_mapping,
            val_mapping=val_mapping,
            task="vehicle_classification",
            spectral_processing=self.config.get('spectral_processing', {}),
            is_training=False
        )
        
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
        
        return train_tf_dataset, val_tf_dataset, train_dataset.num_classes
    
    def _create_model(self, input_shape, num_classes):
        """Create and compile the model."""
        # Determine model type based on input shape
        if len(input_shape) == 1:
            # 1D data (raw time series)
            self.model = create_simple_1d_cnn_model(
                input_shape=input_shape,
                num_classes=num_classes,
                model_name=self.model_name
            )
        else:
            # 2D data (spectrogram)
            self.model = create_seismic_cnn_model(
                input_shape=input_shape,
                num_classes=num_classes,
                model_name=self.model_name
            )
        
        # Compile model
        self.model = compile_model(self.model, learning_rate=0.001)
        
        # Log model architecture
        self.tensorboard_logger.log_model_architecture(self.model)
        
        # Save model summary
        model_summary = self.model.summary()
        with open(os.path.join(self.log_dirs['logs_dir'], 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save parameter count
        param_counts = count_parameters(self.model)
        with open(os.path.join(self.log_dirs['logs_dir'], 'model_parameters.json'), 'w') as f:
            json.dump(param_counts, f, indent=2)
        
        print(f"Model created with {param_counts['total']} total parameters")
        return self.model
    
    def _create_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Model checkpointing
        checkpoint_path = os.path.join(self.log_dirs['models_dir'], 'best_model.h5')
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Custom TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dirs['tensorboard_dir'],
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self, epochs=50, validation_freq=1):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            validation_freq: Frequency of validation during training
        """
        print("Setting up datasets...")
        train_dataset, val_dataset, num_classes = self._setup_datasets()
        
        # Get input shape from a sample
        for batch_data, batch_labels in train_dataset.take(1):
            input_shape = batch_data.shape[1:]  # Remove batch dimension
            break
        
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        
        # Create model
        print("Creating model...")
        self._create_model(input_shape, num_classes)
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Training configuration
        training_config = {
            'epochs': epochs,
            'validation_data': val_dataset,
            'callbacks': callbacks,
            'verbose': 1
        }
        
        # Save training configuration
        with open(os.path.join(self.log_dirs['logs_dir'], 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)
        
        print("Starting training...")
        print(f"Training configuration: {training_config}")
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        history_dict = history.history
        with open(os.path.join(self.log_dirs['logs_dir'], 'training_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print("Training completed!")
        return history
    
    def evaluate(self, test_dataset=None):
        """
        Evaluate the model on validation/test data.
        
        Args:
            test_dataset: Optional test dataset (uses validation if None)
        """
        if test_dataset is None:
            # Use validation dataset
            _, test_dataset, _ = self._setup_datasets()
        
        print("Evaluating model...")
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for batch_data, batch_labels in test_dataset:
            predictions = self.model.predict(batch_data, verbose=0)
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predictions)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        test_loss, test_accuracy, test_top_k_acc = self.model.evaluate(test_dataset, verbose=1)
        
        # Log metrics to TensorBoard
        self.tensorboard_logger.log_validation_metrics({
            'loss': test_loss,
            'accuracy': test_accuracy,
            'top_k_accuracy': test_top_k_acc
        }, step=0)
        
        # Log confusion matrix
        self.tensorboard_logger.log_confusion_matrix(y_true, y_pred, step=0, title="Final Confusion Matrix")
        
        # Log classification report
        self.tensorboard_logger.log_classification_report(y_true, y_pred, step=0, title="Final Classification Report")
        
        # Log sample predictions
        self.tensorboard_logger.log_sample_predictions(
            batch_data[:5], y_true[:5], y_pred[:5], step=0, num_samples=5
        )
        
        # Save evaluation results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_top_k_accuracy': float(test_top_k_acc),
            'num_samples': len(y_true)
        }
        
        with open(os.path.join(self.log_dirs['logs_dir'], 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results: {evaluation_results}")
        return evaluation_results
    
    def save_model_for_tflite_micro(self):
        """
        Save the model in TensorFlow Lite Micro compatible format.
        """
        print("Saving model for TensorFlow Lite Micro...")
        
        # Save the full model first
        model_path = os.path.join(self.log_dirs['models_dir'], 'full_model.h5')
        self.model.save(model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize for size (important for microcontrollers)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set target specs for microcontrollers
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save TensorFlow Lite model
        tflite_path = os.path.join(self.log_dirs['models_dir'], 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save model size information
        model_size = len(tflite_model)
        size_info = {
            'tflite_model_size_bytes': model_size,
            'tflite_model_size_kb': model_size / 1024,
            'tflite_model_size_mb': model_size / (1024 * 1024)
        }
        
        with open(os.path.join(self.log_dirs['logs_dir'], 'model_size_info.json'), 'w') as f:
            json.dump(size_info, f, indent=2)
        
        print(f"TensorFlow Lite model saved to: {tflite_path}")
        print(f"Model size: {size_info['tflite_model_size_kb']:.2f} KB")
        
        return tflite_path, size_info
    
    def save_experiment_summary(self):
        """Save a comprehensive experiment summary."""
        summary = {
            'experiment_id': self.experiment_id,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'class_names': self.class_names,
            'log_dirs': self.log_dirs
        }
        
        # Add model info if available
        if self.model is not None:
            summary['model_parameters'] = count_parameters(self.model)
            summary['model_input_shape'] = self.model.input_shape
            summary['model_output_shape'] = self.model.output_shape
        
        summary_path = os.path.join(self.log_dirs['experiment_dir'], 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment summary saved to: {summary_path}")
        return summary
    
    def close(self):
        """Close TensorBoard logger and cleanup."""
        if self.tensorboard_logger:
            self.tensorboard_logger.close()


def main():
    """Main training script."""
    # Configuration
    config_path = 'demo_dataset_config.yaml'
    model_name = "SeismicCNN"
    
    # Create trainer
    trainer = SeismicModelTrainer(config_path, model_name)
    
    try:
        # Train the model
        history = trainer.train(epochs=50)
        
        # Evaluate the model
        evaluation_results = trainer.evaluate()
        
        # Save model for TensorFlow Lite Micro
        tflite_path, size_info = trainer.save_model_for_tflite_micro()
        
        # Save experiment summary
        summary = trainer.save_experiment_summary()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Experiment ID: {trainer.experiment_id}")
        print(f"Model saved to: {trainer.log_dirs['models_dir']}")
        print(f"TensorBoard logs: {trainer.log_dirs['tensorboard_dir']}")
        print(f"TensorFlow Lite model: {tflite_path}")
        print(f"Model size: {size_info['tflite_model_size_kb']:.2f} KB")
        print("="*50)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        trainer.close()


if __name__ == "__main__":
    main()
