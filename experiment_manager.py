#!/usr/bin/env python3
"""
Experiment Manager - Utility script for managing and viewing experiments
"""

import os
import sys
from experiment_utils import (
    list_experiments, load_experiment_metadata, print_experiment_summary,
    get_experiment_paths
)


def main():
    """Main function for experiment management"""
    if len(sys.argv) < 2:
        print("Experiment Manager")
        print("=" * 50)
        print("Usage:")
        print("  python experiment_manager.py list                    # List all experiments")
        print("  python experiment_manager.py summary <experiment_id> # Show experiment summary")
        print("  python experiment_manager.py tensorboard <experiment_id> # Show TensorBoard command")
        print("  python experiment_manager.py compare <exp1> <exp2>   # Compare two experiments")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_all_experiments()
    elif command == "summary":
        if len(sys.argv) < 3:
            print("Error: Please provide experiment ID")
            return
        experiment_id = sys.argv[2]
        print_experiment_summary(experiment_id)
    elif command == "tensorboard":
        if len(sys.argv) < 3:
            print("Error: Please provide experiment ID")
            return
        experiment_id = sys.argv[2]
        show_tensorboard_command(experiment_id)
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Error: Please provide two experiment IDs")
            return
        exp1_id = sys.argv[2]
        exp2_id = sys.argv[3]
        compare_experiments(exp1_id, exp2_id)
    else:
        print(f"Unknown command: {command}")


def list_all_experiments():
    """List all available experiments"""
    experiments = list_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    print("=" * 80)
    
    for i, exp_id in enumerate(experiments, 1):
        try:
            metadata = load_experiment_metadata(exp_id)
            print(f"{i:2d}. {exp_id}")
            print(f"    Model: {metadata['model_info']['type']}")
            print(f"    Classes: {metadata['model_info']['num_classes']}")
            print(f"    Parameters: {int(metadata['model_info']['total_params']):,}")
            print(f"    Timestamp: {metadata['timestamp']}")
            print()
        except Exception as e:
            print(f"{i:2d}. {exp_id} (Error loading metadata: {e})")
            print()


def show_tensorboard_command(experiment_id):
    """Show TensorBoard command for an experiment"""
    try:
        paths = get_experiment_paths(experiment_id)
        logs_dir = paths['logs_dir']
        
        print(f"TensorBoard Command for {experiment_id}:")
        print("=" * 60)
        print(f"tensorboard --logdir {logs_dir}")
        print()
        print("Then open http://localhost:6006 in your browser")
        print()
        print("Available tabs in TensorBoard:")
        print("- SCALARS: Training metrics (loss, accuracy)")
        print("- IMAGES: Confusion matrices")
        print("- HISTOGRAMS: Model weights")
        print("- TEXT: Experiment metadata")
        
    except Exception as e:
        print(f"Error: {e}")


def compare_experiments(exp1_id, exp2_id):
    """Compare two experiments"""
    try:
        print(f"Comparing Experiments:")
        print("=" * 80)
        print(f"Experiment 1: {exp1_id}")
        print(f"Experiment 2: {exp2_id}")
        print("=" * 80)
        
        # Load metadata for both experiments
        metadata1 = load_experiment_metadata(exp1_id)
        metadata2 = load_experiment_metadata(exp2_id)
        
        # Compare model info
        print("MODEL COMPARISON:")
        print("-" * 40)
        print(f"{'Metric':<25} {'Exp1':<20} {'Exp2':<20}")
        print("-" * 65)
        print(f"{'Model Type':<25} {metadata1['model_info']['type']:<20} {metadata2['model_info']['type']:<20}")
        print(f"{'Parameters':<25} {int(metadata1['model_info']['total_params']):<20,} {int(metadata2['model_info']['total_params']):<20,}")
        print(f"{'Input Shape':<25} {str(metadata1['model_info']['input_shape']):<20} {str(metadata2['model_info']['input_shape']):<20}")
        print(f"{'Classes':<25} {metadata1['model_info']['num_classes']:<20} {metadata2['model_info']['num_classes']:<20}")
        
        # Compare training config
        print("\nTRAINING CONFIG COMPARISON:")
        print("-" * 40)
        print(f"{'Parameter':<25} {'Exp1':<20} {'Exp2':<20}")
        print("-" * 65)
        
        config1 = metadata1['training_config']
        config2 = metadata2['training_config']
        
        all_keys = set(config1.keys()) | set(config2.keys())
        for key in sorted(all_keys):
            val1 = config1.get(key, 'N/A')
            val2 = config2.get(key, 'N/A')
            print(f"{key:<25} {str(val1):<20} {str(val2):<20}")
        
        # Show TensorBoard comparison command
        print("\nTENSORBOARD COMPARISON:")
        print("-" * 40)
        paths1 = get_experiment_paths(exp1_id)
        paths2 = get_experiment_paths(exp2_id)
        print(f"tensorboard --logdir_spec={exp1_id}:{paths1['logs_dir']},{exp2_id}:{paths2['logs_dir']}")
        print("This will show both experiments side-by-side in TensorBoard")
        
    except Exception as e:
        print(f"Error comparing experiments: {e}")


if __name__ == "__main__":
    main()
