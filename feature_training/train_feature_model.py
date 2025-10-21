#!/usr/bin/env python3
"""
Main training script for feature-based seismic classification model.

This script implements the complete pipeline:
1. Load and preprocess data
2. Extract features from raw seismic data
3. Normalize features using training statistics
4. Calculate background energy threshold
5. Train a compact neural network
6. Evaluate end-to-end performance with background filtering
"""

import os
import sys
import yaml
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import our utilities
from feature_utils import (
    get_mapped_dataset, load_data, reshape_data_to_2d, print_data_shape,
    extract_all_features_from_data, compute_feature_statistics, 
    normalize_features, features_dict_to_array, save_feature_statistics,
    remove_background_from_training_data, 
    balance_classes_by_undersampling, 
    undersample_background
)
from train_utils import (
    calculate_background_energy_threshold, create_dense_model, compile_model,
    train_model, plot_training_history, save_training_config, save_model_info,
    create_experiment_directory, get_model_size_kb
)
from test_utils import (
    encode_labels, evaluate_end_to_end, plot_confusion_matrix,
    save_evaluation_report, save_evaluation_metrics, debug_class_distribution,
    debug_model_predictions
)


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("FEATURE-BASED SEISMIC CLASSIFICATION TRAINING")
    print("="*80)
    
    # Load configuration
    config_path = 'train_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuration loaded from: {config_path}")
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(config['output']['experiment_dir'])
    
    # Save configuration to experiment directory
    save_training_config(config, experiment_dir)
    
    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Get dataset mappings
    train_mapping, val_mapping = get_mapped_dataset(config)
    
    
    print("After filtering by max distance:")
    print("Train mapping:")
    for vehicle, file_paths in train_mapping.items():
        print(f"  {vehicle}: {len(file_paths)}")
    print("Val mapping:")
    for vehicle, file_paths in val_mapping.items():
        print(f"  {vehicle}: {len(file_paths)}")
    
    # Load raw data
    print("\nLoading raw data...")
    train_data = load_data(train_mapping)
    val_data = load_data(val_mapping)
    print("✓ Raw data loaded successfully!")
    
    # Print data shapes
    print("\nTrain data shapes:")
    print_data_shape(train_data)
    print("Val data shapes:")
    print_data_shape(val_data)
    
    # Reshape data to 2D
    print("\nReshaping data to 2D...")
    train_data = reshape_data_to_2d(train_data)
    val_data = reshape_data_to_2d(val_data)
    print("✓ Data reshaped successfully!")
    
    print("\nReshaped data shapes:")
    print("Train data shapes:")
    print_data_shape(train_data)
    print("Val data shapes:")
    print_data_shape(val_data)
    
    # Remove background samples from training data to avoid class imbalance
    # print("\n" + "="*70)
    # print("STEP 1.5: REMOVE BACKGROUND FROM TRAINING DATA")
    # print("="*70)
    # train_data = remove_background_from_training_data(train_data, verbose=True)
    # print("\nFiltered training data shapes:")
    # print_data_shape(train_data)
    
    # ============================================================================
    # FEATURE EXTRACTION
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*70)
    
    # Extract features from training data (filtered, no background)
    print("Extracting features from training data (no background)...")
    train_features = extract_all_features_from_data(train_data, verbose=True)
    
    # Extract features from validation data (includes background)
    print("\nExtracting features from validation data...")
    val_features = extract_all_features_from_data(val_data, verbose=True)
    
    # Filter features to only include those specified in config
    # print("\nFiltering features to specified subset...")
    # features_to_use = config['features_to_use']
    # print(f"Features to use: {features_to_use}")
    
    # train_features = filter_features_by_config(train_features, features_to_use)
    # val_features = filter_features_by_config(val_features, features_to_use)
    
    # print(f"✓ Filtered training features: {list(train_features.keys())}")
    # print(f"✓ Filtered validation features: {list(val_features.keys())}")
    
    # ============================================================================
    # FEATURE NORMALIZATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 3: FEATURE NORMALIZATION")
    print("="*70)
    
    # Compute normalization statistics from training data only (no background)
    print("Computing feature statistics from training data...")
    feature_statistics = compute_feature_statistics(train_features, verbose=True)
    
    # Save feature statistics
    stats_path = os.path.join(experiment_dir, 'feature_statistics.json')
    save_feature_statistics(feature_statistics, stats_path)
    
    # Normalize features using training statistics
    print("\nNormalizing training features...")
    train_features_normalized = normalize_features(
        train_features, 
        feature_statistics, 
        verbose=True
    )
    
    print("\nNormalizing validation features...")
    val_features_normalized = normalize_features(
        val_features, 
        feature_statistics,  # Use TRAINING statistics (no data leakage!)
        verbose=True
    )
    
    # ============================================================================
    # BACKGROUND ENERGY THRESHOLD CALCULATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 4: BACKGROUND ENERGY THRESHOLD")
    print("="*70)
    
    # For background threshold calculation, we need the original training data with background
    # Load original training data again for threshold calculation
    print("Loading original training data for background threshold calculation...")
    train_data_original = load_data(train_mapping)
    train_data_original = reshape_data_to_2d(train_data_original)
    train_features_original = extract_all_features_from_data(train_data_original, verbose=False)
    train_features_original_normalized = normalize_features(
        train_features_original, 
        feature_statistics, 
        verbose=False
    )
    
    # Calculate background energy threshold from original training data (with background)
    background_threshold = calculate_background_energy_threshold(train_features_original_normalized, config)
    
    # Save threshold
    threshold_path = os.path.join(experiment_dir, 'background_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': background_threshold}, f, indent=2)
    print(f"✓ Background threshold saved to: {threshold_path}")
    
    # ============================================================================
    # PREPARE DATA FOR NEURAL NETWORK
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 5: PREPARE DATA FOR NEURAL NETWORK")
    print("="*70)
    
    # Convert features to arrays
    X_train, feature_names = features_dict_to_array(train_features_normalized)
    X_val, _ = features_dict_to_array(val_features_normalized, feature_order=feature_names)
    # Encode labels
    class_names = config['vehicle_classification']['included_classes']
    y_train = encode_labels(train_features_normalized['vehicle_labels'], class_names)
    y_val = encode_labels(val_features_normalized['vehicle_labels'], class_names)

    # Balance classes by undersampling
    # breakpoint()
    X_train, y_train = undersample_background(X_train, y_train, target_ratio=1.0, random_state=42, verbose=True)
    
    print(f"✓ Training data shape: {X_train.shape}")
    print(f"✓ Validation data shape: {X_val.shape}")
    print(f"✓ Feature names: {feature_names}")
    print(f"✓ Number of classes: {len(class_names)}")
    
    # Debug class distribution and one-hot encoding
    # debug_class_distribution(train_features_normalized, class_names, "TRAINING DATA")
    # debug_class_distribution(val_features_normalized, class_names, "VALIDATION DATA")
    # breakpoint()
    
    # ============================================================================
    # MODEL CREATION AND COMPILATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 6: MODEL CREATION")
    print("="*70)
    
    # Create model
    input_dim = X_train.shape[1]
    num_classes = len(class_names)
    # breakpoint()
    
    model = create_dense_model(input_dim, num_classes, config)
    model = compile_model(model, config)
    
    # Check model size
    model_size_kb = get_model_size_kb(model)
    print(f"✓ Model created with {model.count_params():,} parameters")
    print(f"✓ Model size: {model_size_kb:.2f} KB")
    
    if model_size_kb > 600:
        print(f"⚠️  WARNING: Model size ({model_size_kb:.2f} KB) exceeds 600 KB target!")
    else:
        print(f"✓ Model size is within 600 KB target")
    
    # Save model info
    save_model_info(model, experiment_dir)
    
    # ============================================================================
    # MODEL TRAINING
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 7: MODEL TRAINING")
    print("="*70)
    
    # Train model
    print("Starting model training...")
    # breakpoint()
    history = train_model(model, X_train, y_train, X_val, y_val, config, experiment_dir)
    
    # Plot training history
    history_path = os.path.join(experiment_dir, 'plots', 'training_history.png')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    plot_training_history(history, history_path)
    
    print("✓ Model training completed!")
    
    # Debug model predictions before end-to-end evaluation
    debug_model_predictions(model, X_val, class_names, "MODEL PREDICTIONS (Before Background Filter)")
    
    # ============================================================================
    # END-TO-END EVALUATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("STEP 8: END-TO-END EVALUATION")
    print("="*70)
    
    # Evaluate end-to-end pipeline
    evaluation_results = evaluate_end_to_end(
        model, X_val, y_val, val_features_normalized, 
        background_threshold, class_names
    )
    
    # Save evaluation results
    metrics_path = os.path.join(experiment_dir, 'logs', 'evaluation_metrics.json')
    save_evaluation_metrics(evaluation_results, metrics_path)
    
    report_path = os.path.join(experiment_dir, 'logs', 'evaluation_report.txt')
    save_evaluation_report(evaluation_results, class_names, report_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(experiment_dir, 'plots', 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(evaluation_results['confusion_matrix']), 
        class_names, 
        cm_path
    )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Overall Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"  F1 Score (Macro): {evaluation_results['f1_macro']:.4f}")
    print(f"  Recall (Macro): {evaluation_results['recall_macro']:.4f}")
    print(f"  Background Threshold: {evaluation_results['threshold']:.6f}")
    print(f"  Model Size: {model_size_kb:.2f} KB")
    
    print(f"\n📁 EXPERIMENT OUTPUTS:")
    print(f"  Experiment Directory: {experiment_dir}")
    print(f"  Best Model: {experiment_dir}/models/best_model.keras")
    print(f"  Feature Statistics: {experiment_dir}/feature_statistics.json")
    print(f"  Background Threshold: {experiment_dir}/background_threshold.json")
    print(f"  Training History: {experiment_dir}/plots/training_history.png")
    print(f"  Confusion Matrix: {experiment_dir}/plots/confusion_matrix.png")
    print(f"  Evaluation Report: {experiment_dir}/logs/evaluation_report.txt")
    
    print(f"\n✅ All outputs saved successfully!")


if __name__ == "__main__":
    main()
