#!/usr/bin/env python3
"""
Example script demonstrating how to use the seismic sensor training system.

This script shows how to:
1. Train a CNN model on seismic sensor data
2. Save the model in TensorFlow Lite Micro format
3. Use TensorBoard for monitoring training
4. Generate comprehensive evaluation metrics
"""

import os
import sys
from train_and_eval import SeismicModelTrainer


def main():
    """
    Main example function demonstrating the training pipeline.
    """
    print("="*60)
    print("SEISMIC SENSOR CNN TRAINING EXAMPLE")
    print("="*60)
    
    # Configuration
    config_path = 'demo_dataset_config.yaml'
    model_name = "SeismicCNN_Example"
    
    print(f"Configuration file: {config_path}")
    print(f"Model name: {model_name}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file '{config_path}' not found!")
        print("Please make sure the demo_dataset_config.yaml file exists.")
        return
    
    try:
        # Create trainer
        print("\n1. Initializing trainer...")
        trainer = SeismicModelTrainer(config_path)
        
        print(f"   Experiment ID: {trainer.experiment_id}")
        print(f"   Log directories created in: {trainer.log_dirs['experiment_dir']}")
        
        # Train the model
        print("\n2. Starting training...")
        print("   This will train the CNN model on your seismic sensor data.")
        print("   Training progress will be logged to TensorBoard.")
        
        # history = trainer.train(epochs=30)  # Reduced epochs for example
        
        # print("   Training completed successfully!")
        
        # # Evaluate the model
        # print("\n3. Evaluating model...")
        # evaluation_results = trainer.evaluate()
        
        # print(f"   Final accuracy: {evaluation_results['test_accuracy']:.4f}")
        # print(f"   Final loss: {evaluation_results['test_loss']:.4f}")
        
        # # Save model for TensorFlow Lite Micro
        # print("\n4. Saving model for TensorFlow Lite Micro...")
        # tflite_path, size_info = trainer.save_model_for_tflite_micro()
        
        # print(f"   TensorFlow Lite model saved to: {tflite_path}")
        # print(f"   Model size: {size_info['tflite_model_size_kb']:.2f} KB")
        
        # # Save experiment summary
        # print("\n5. Saving experiment summary...")
        # summary = trainer.save_experiment_summary()
        
        # # Display results
        # print("\n" + "="*60)
        # print("TRAINING COMPLETED SUCCESSFULLY!")
        # print("="*60)
        # print(f"Experiment ID: {trainer.experiment_id}")
        # print(f"Model accuracy: {evaluation_results['test_accuracy']:.4f}")
        # print(f"Model size: {size_info['tflite_model_size_kb']:.2f} KB")
        # print(f"TensorBoard logs: {trainer.log_dirs['tensorboard_dir']}")
        # print(f"Model files: {trainer.log_dirs['models_dir']}")
        # print(f"Experiment summary: {trainer.log_dirs['experiment_dir']}/experiment_summary.json")
        
        # print("\nTo view training progress:")
        # print(f"  tensorboard --logdir {trainer.log_dirs['tensorboard_dir']}")
        
        # print("\nTo use the trained model:")
        # print(f"  - Full model: {trainer.log_dirs['models_dir']}/full_model.h5")
        # print(f"  - TensorFlow Lite: {tflite_path}")
        
        # print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Training failed with error: {e}")
        print("Please check your configuration and data paths.")
        return 1
    
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.close()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
