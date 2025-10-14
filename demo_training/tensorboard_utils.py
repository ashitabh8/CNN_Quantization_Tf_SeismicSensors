import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import os


class TensorBoardLogger:
    """
    Custom TensorBoard logger for seismic sensor model training.
    Handles logging of metrics, confusion matrices, and model analysis.
    """
    
    def __init__(self, log_dir, class_names=None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            class_names: List of class names for confusion matrix labels
        """
        self.log_dir = log_dir
        self.class_names = class_names or [f"Class_{i}" for i in range(10)]
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
    def log_scalar(self, name, value, step):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Step number
        """
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)
        self.writer.flush()
    
    def log_histogram(self, name, values, step):
        """
        Log a histogram to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Step number
        """
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
        self.writer.flush()
    
    def log_confusion_matrix(self, y_true, y_pred, step, title="Confusion Matrix"):
        """
        Log confusion matrix to TensorBoard.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (one-hot encoded)
            step: Step number
            title: Title for the confusion matrix
        """
        # Convert one-hot to class indices
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.image(title, image, step=step)
        
        plt.close()
        self.writer.flush()
    
    def log_classification_report(self, y_true, y_pred, step, title="Classification Report"):
        """
        Log classification report as text to TensorBoard.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (one-hot encoded)
            step: Step number
            title: Title for the report
        """
        # Convert one-hot to class indices
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                    target_names=self.class_names, 
                                    output_dict=False)
        
        # Log as text
        with self.writer.as_default():
            tf.summary.text(title, report, step=step)
        self.writer.flush()
    
    def log_model_architecture(self, model, step=0):
        """
        Log model architecture to TensorBoard.
        
        Args:
            model: Keras model
            step: Step number
        """
        # Get model summary
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Log model summary
        with self.writer.as_default():
            tf.summary.text("Model Architecture", summary, step=step)
        self.writer.flush()
    
    def log_learning_rate(self, lr, step):
        """
        Log learning rate to TensorBoard.
        
        Args:
            lr: Learning rate value
            step: Step number
        """
        self.log_scalar("Learning Rate", lr, step)
    
    def log_training_metrics(self, metrics_dict, step):
        """
        Log training metrics to TensorBoard.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step number
        """
        for metric_name, value in metrics_dict.items():
            self.log_scalar(f"Training/{metric_name}", value, step)
    
    def log_validation_metrics(self, metrics_dict, step):
        """
        Log validation metrics to TensorBoard.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step number
        """
        for metric_name, value in metrics_dict.items():
            self.log_scalar(f"Validation/{metric_name}", value, step)
    
    def log_loss_curves(self, train_loss, val_loss, step):
        """
        Log loss curves to TensorBoard.
        
        Args:
            train_loss: Training loss value
            val_loss: Validation loss value
            step: Step number
        """
        self.log_scalar("Loss/Train", train_loss, step)
        self.log_scalar("Loss/Validation", val_loss, step)
    
    def log_accuracy_curves(self, train_acc, val_acc, step):
        """
        Log accuracy curves to TensorBoard.
        
        Args:
            train_acc: Training accuracy value
            val_acc: Validation accuracy value
            step: Step number
        """
        self.log_scalar("Accuracy/Train", train_acc, step)
        self.log_scalar("Accuracy/Validation", val_acc, step)
    
    def log_gradients(self, model, step):
        """
        Log gradient histograms for all trainable variables.
        
        Args:
            model: Keras model
            step: Step number
        """
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                self.log_histogram(f"Gradients/{layer.name}_kernel", 
                                 layer.kernel, step)
            if hasattr(layer, 'bias') and layer.bias is not None:
                self.log_histogram(f"Gradients/{layer.name}_bias", 
                                 layer.bias, step)
    
    def log_weights(self, model, step):
        """
        Log weight histograms for all trainable variables.
        
        Args:
            model: Keras model
            step: Step number
        """
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                self.log_histogram(f"Weights/{layer.name}_kernel", 
                                 layer.kernel, step)
            if hasattr(layer, 'bias') and layer.bias is not None:
                self.log_histogram(f"Weights/{layer.name}_bias", 
                                 layer.bias, step)
    
    def log_sample_predictions(self, x_batch, y_true, y_pred, step, num_samples=5):
        """
        Log sample predictions with confidence scores.
        
        Args:
            x_batch: Input batch
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            step: Step number
            num_samples: Number of samples to log
        """
        # Get top predictions
        top_pred_indices = np.argsort(y_pred, axis=1)[:, -2:]  # Top 2 predictions
        
        # Create text summary
        summary_text = "Sample Predictions:\n\n"
        for i in range(min(num_samples, len(x_batch))):
            true_class = np.argmax(y_true[i])
            pred_class = np.argmax(y_pred[i])
            confidence = y_pred[i][pred_class]
            
            summary_text += f"Sample {i+1}:\n"
            summary_text += f"  True: {self.class_names[true_class]}\n"
            summary_text += f"  Predicted: {self.class_names[pred_class]} (conf: {confidence:.3f})\n"
            summary_text += f"  Top 2: {[self.class_names[idx] for idx in top_pred_indices[i]]}\n\n"
        
        with self.writer.as_default():
            tf.summary.text("Sample Predictions", summary_text, step=step)
        self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


def create_experiment_id(model_name, timestamp=None):
    """
    Create a unique experiment ID.
    
    Args:
        model_name: Name of the model
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        str: Unique experiment ID
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{date_str}"


def setup_experiment_logging(experiment_id, base_dir="experiments"):
    """
    Setup experiment logging directory structure.
    
    Args:
        experiment_id: Unique experiment ID
        base_dir: Base directory for experiments
    
    Returns:
        dict: Dictionary with log directory paths
    """
    experiment_dir = os.path.join(base_dir, experiment_id)
    
    log_dirs = {
        'experiment_dir': experiment_dir,
        'tensorboard_dir': os.path.join(experiment_dir, 'tensorboard'),
        'models_dir': os.path.join(experiment_dir, 'models'),
        'plots_dir': os.path.join(experiment_dir, 'plots'),
        'logs_dir': os.path.join(experiment_dir, 'logs')
    }
    
    # Create directories
    for dir_path in log_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return log_dirs
