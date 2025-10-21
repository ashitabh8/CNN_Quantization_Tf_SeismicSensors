import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def apply_background_filter(features_dict, threshold):
    """
    Classify samples below energy threshold as background.
    
    Args:
        features_dict: dict with 'total_power' and 'vehicle_labels' keys
        threshold: Energy threshold for background classification
    
    Returns:
        tuple: (filtered_predictions, filtered_indices, background_indices)
            - filtered_predictions: Predictions for samples above threshold
            - filtered_indices: Indices of samples above threshold
            - background_indices: Indices of samples classified as background
    """
    total_power = features_dict['total_power']
    
    # Find samples above threshold (non-background)
    above_threshold_mask = total_power >= threshold
    below_threshold_mask = total_power < threshold
    
    # Get indices
    filtered_indices = np.where(above_threshold_mask)[0]
    background_indices = np.where(below_threshold_mask)[0]
    
    print(f"Background filter applied:")
    print(f"  Samples above threshold: {len(filtered_indices)}")
    print(f"  Samples below threshold (background): {len(background_indices)}")
    print(f"  Threshold: {threshold:.6f}")
    
    return filtered_indices, background_indices


def encode_labels(labels, class_names):
    """
    Convert string labels to one-hot encoding.
    
    Args:
        labels: Array of string labels
        class_names: List of class names in order
    
    Returns:
        np.array: One-hot encoded labels
    """
    # Create mapping from class name to index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Convert labels to indices
    label_indices = np.array([class_to_idx[label] for label in labels])
    
    # Convert to one-hot
    one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=len(class_names))
    
    return one_hot


def decode_predictions(predictions, class_names):
    """
    Convert one-hot predictions back to class names.
    
    Args:
        predictions: One-hot encoded predictions or probability array
        class_names: List of class names in order
    
    Returns:
        np.array: Array of predicted class names
    """
    if predictions.ndim > 1:
        # Convert from one-hot to class indices
        predicted_indices = np.argmax(predictions, axis=1)
    else:
        predicted_indices = predictions
    
    # Convert indices to class names
    predicted_labels = np.array([class_names[idx] for idx in predicted_indices])
    
    return predicted_labels


def evaluate_end_to_end(model, X_val, y_val, features_dict, threshold, class_names):
    """
    Evaluate end-to-end pipeline with background filtering.
    
    Args:
        model: Trained tf.keras.Model
        X_val: Validation features array
        y_val: Validation labels (one-hot encoded)
        features_dict: Features dict with 'total_power' and 'vehicle_labels'
        threshold: Background energy threshold
        class_names: List of class names
    
    Returns:
        dict: Evaluation metrics and results
    """
    print("\n" + "="*70)
    print("END-TO-END EVALUATION")
    print("="*70)
    
    # Step 1: Apply background filter
    filtered_indices, background_indices = apply_background_filter(features_dict, threshold)
    
    # Step 2: Get true labels for all samples
    true_labels = features_dict['vehicle_labels']
    true_labels_encoded = encode_labels(true_labels, class_names)
    
    # Step 3: Initialize predictions array
    n_samples = len(true_labels)
    n_classes = len(class_names)
    all_predictions = np.zeros((n_samples, n_classes))
    
    # Step 4: Set background predictions for samples below threshold
    background_class_idx = class_names.index('background')
    all_predictions[background_indices, background_class_idx] = 1.0
    
    # Step 5: Run neural network on samples above threshold
    if len(filtered_indices) > 0:
        X_filtered = X_val[filtered_indices]
        nn_predictions = model.predict(X_filtered, verbose=0)
        all_predictions[filtered_indices] = nn_predictions
    
    # Step 6: Convert to class predictions
    predicted_labels = decode_predictions(all_predictions, class_names)
    true_labels_decoded = decode_predictions(true_labels_encoded, class_names)
    
    # Step 7: Calculate metrics
    accuracy = accuracy_score(true_labels_decoded, predicted_labels)
    f1_macro = f1_score(true_labels_decoded, predicted_labels, average='macro')
    f1_per_class = f1_score(true_labels_decoded, predicted_labels, average=None)
    recall_macro = recall_score(true_labels_decoded, predicted_labels, average='macro')
    recall_per_class = recall_score(true_labels_decoded, predicted_labels, average=None)
    
    # Step 8: Generate confusion matrix
    cm = confusion_matrix(true_labels_decoded, predicted_labels, labels=class_names)
    
    # Step 9: Generate classification report
    report = classification_report(
        true_labels_decoded, 
        predicted_labels, 
        labels=class_names,
        output_dict=True
    )
    
    # Step 10: Compile results
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_per_class': f1_per_class.tolist(),
        'recall_macro': float(recall_macro),
        'recall_per_class': recall_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'n_background_filtered': len(background_indices),
        'n_nn_predictions': len(filtered_indices),
        'threshold': float(threshold)
    }
    
    # Print summary
    print(f"\nEnd-to-End Results:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  Recall (Macro): {recall_macro:.4f}")
    print(f"  Background samples filtered: {len(background_indices)}")
    print(f"  NN predictions made: {len(filtered_indices)}")
    
    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Create and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")


def save_evaluation_report(results, class_names, save_path):
    """
    Save detailed evaluation report to text file.
    
    Args:
        results: Results dictionary from evaluate_end_to_end
        class_names: List of class names
        save_path: Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("FEATURE-BASED MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("END-TO-END PIPELINE RESULTS:\n")
        f.write(f"  Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  F1 Score (Macro): {results['f1_macro']:.4f}\n")
        f.write(f"  Recall (Macro): {results['recall_macro']:.4f}\n")
        f.write(f"  Background Energy Threshold: {results['threshold']:.6f}\n")
        f.write(f"  Samples filtered as background: {results['n_background_filtered']}\n")
        f.write(f"  Samples processed by NN: {results['n_nn_predictions']}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:12s}: F1={results['f1_per_class'][i]:.4f}, "
                   f"Recall={results['recall_per_class'][i]:.4f}\n")
        
        f.write("\nCONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        f.write("True\\Pred".ljust(12))
        for name in class_names:
            f.write(f"{name:>8s}")
        f.write("\n")
        
        for i, true_class in enumerate(class_names):
            f.write(f"{true_class:12s}")
            for j in range(len(class_names)):
                f.write(f"{results['confusion_matrix'][i][j]:>8d}")
            f.write("\n")
        
        f.write("\nDETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        
        # Write per-class metrics from sklearn report
        report = results['classification_report']
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
    
    print(f"✓ Evaluation report saved to: {save_path}")


def save_evaluation_metrics(results, save_path):
    """
    Save evaluation metrics as JSON for programmatic access.
    
    Args:
        results: Results dictionary from evaluate_end_to_end
        save_path: Path to save the metrics JSON
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Evaluation metrics saved to: {save_path}")


def debug_class_distribution(features_dict, class_names, title="Class Distribution"):
    """
    Debug function to print class distribution and one-hot encoding verification.
    
    Args:
        features_dict: dict with 'vehicle_labels' key
        class_names: List of class names
        title: Title for the debug output
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Count class distribution
    labels = features_dict['vehicle_labels']
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        print(f"  {label:12s}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Verify one-hot encoding
    print("\nOne-hot encoding verification:")
    encoded = encode_labels(labels, class_names)
    print(f"  Input labels shape: {labels.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Expected classes: {len(class_names)}")
    print(f"  Class names: {class_names}")
    
    # Check if all classes are present
    present_classes = set(unique_labels)
    expected_classes = set(class_names)
    missing_classes = expected_classes - present_classes
    extra_classes = present_classes - expected_classes
    
    if missing_classes:
        print(f"  ⚠️  Missing classes: {missing_classes}")
    if extra_classes:
        print(f"  ⚠️  Extra classes: {extra_classes}")
    
    # Verify one-hot encoding is correct
    decoded = decode_predictions(encoded, class_names)
    encoding_correct = np.array_equal(labels, decoded)
    print(f"  One-hot encoding correct: {encoding_correct}")
    
    if not encoding_correct:
        print("  ⚠️  One-hot encoding mismatch detected!")
        # Show first few mismatches
        mismatches = labels != decoded
        if np.any(mismatches):
            print("  First few mismatches:")
            mismatch_indices = np.where(mismatches)[0][:5]
            for idx in mismatch_indices:
                print(f"    Index {idx}: original='{labels[idx]}', decoded='{decoded[idx]}'")


def debug_model_predictions(model, X_val, class_names, title="Model Predictions Debug"):
    """
    Debug function to analyze model predictions and identify biases.
    
    Args:
        model: Trained model
        X_val: Validation features
        class_names: List of class names
        title: Title for the debug output
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Get model predictions
    predictions = model.predict(X_val, verbose=0)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of samples: {len(predictions)}")
    
    # Analyze prediction distribution
    predicted_classes = np.argmax(predictions, axis=1)
    unique_preds, pred_counts = np.unique(predicted_classes, return_counts=True)
    
    print("\nPredicted class distribution:")
    for class_idx, count in zip(unique_preds, pred_counts):
        class_name = class_names[class_idx]
        percentage = (count / len(predictions)) * 100
        print(f"  {class_name:12s}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Analyze prediction confidence
    max_probs = np.max(predictions, axis=1)
    print(f"\nPrediction confidence statistics:")
    print(f"  Mean confidence: {np.mean(max_probs):.4f}")
    print(f"  Std confidence:  {np.std(max_probs):.4f}")
    print(f"  Min confidence:  {np.min(max_probs):.4f}")
    print(f"  Max confidence:  {np.max(max_probs):.4f}")
    
    # Check for prediction bias
    print(f"\nPrediction bias analysis:")
    for i, class_name in enumerate(class_names):
        class_probs = predictions[:, i]
        mean_prob = np.mean(class_probs)
        print(f"  {class_name:12s}: mean probability = {mean_prob:.4f}")
    
    return predictions
