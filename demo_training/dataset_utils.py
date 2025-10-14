
import yaml
import torch
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np


def read_list_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def load_sample(file_path):
    sample = torch.load(file_path, map_location="cpu", weights_only=False)
    return sample 

def give_count_from_mapping(vehicle_to_file_path_mapping):
    for vehicle, file_paths in vehicle_to_file_path_mapping.items():
        print(f"{vehicle}: {len(file_paths)}")

def filter_samples_by_max_distance(vehicle_to_file_path_mapping, max_distance_m):
    #ignore background
    filtered_mapping = {vehicle: [] for vehicle in vehicle_to_file_path_mapping.keys()}
    from tqdm import tqdm
    
    for vehicle, file_paths in vehicle_to_file_path_mapping.items():
        if vehicle == "background":
            filtered_mapping[vehicle] = file_paths
            continue
        for file_path in tqdm(file_paths, desc=f"Processing {vehicle}", leave=False):
            sample = load_sample(file_path)
            try:
                distance = sample.get('distance')
                if not isinstance(distance, dict):
                    raise ValueError(f"Expected 'distance' to be a dictionary but got {type(distance)}")
                distance_m = distance.get(vehicle, float('nan'))
                if not isinstance(distance_m, (int, float)):
                    raise ValueError(f"Distance value for vehicle '{vehicle}' must be a number, got {type(distance_m)}")
            except Exception as e:
                raise ValueError(f"Error getting distance for vehicle '{vehicle}': {str(e)}")
            if np.isfinite(distance_m) and distance_m <= max_distance_m:
                filtered_mapping[vehicle].append(file_path)
    return filtered_mapping

def filter_single_vehicle(items, vehicle_names):
    """Filter items to only include those with single vehicle names."""
    single_vehicle_items = []
    for item in items:
        is_single, vehicle = is_single_vehicle_item(item, vehicle_names)
        if is_single:
            single_vehicle_items.append(item)
    return single_vehicle_items


def is_single_vehicle_item(item: str, vehicle_names: List[str]) -> Tuple[bool, str]:
    """Check if item contains only one vehicle name and return the vehicle."""
    # Extract just the filename from the full path
    filename = Path(item).name
    matched = [v for v in vehicle_names if re.search(rf"_{re.escape(v)}_", filename, re.IGNORECASE)]
    if len(matched) == 1:
        return True, matched[0]
    return False, ""


def print_shapes(vehicle_to_file_path_mapping):
    # Just pick first vehicle and first file
    vehicle, file_paths = next(iter(vehicle_to_file_path_mapping.items()))
    if file_paths:
        sample = load_sample(file_paths[0])
        # breakpoint()
        print(f"{vehicle}: {sample['data']['shake']['seismic'].shape}")


def is_background_item(item: str, background_run_indices: List[int] = [14, 15]) -> bool:
    """Detect background samples identified by run indices in the filename.
    
    Args:
        item: The filename to check
        background_run_indices: List of run indices that represent background samples
    """
    name = Path(item).name  # operate on basename
    # Match run followed by the specific indices (e.g., run14, run15)
    pattern = r"run(?:" + "|".join(str(i) for i in background_run_indices) + r")(?:_|\.|$)"
    return bool(re.search(pattern, name, flags=re.IGNORECASE))

def create_mapping_vehicle_name_to_file_path(config):
    from tqdm import tqdm
    
    included_classes = config['vehicle_classification']['included_classes']
    vehicle_names = config['vehicle_classification']['class_names']
    train_index_file = config['vehicle_classification']['train_index_file']
    val_index_file = config['vehicle_classification']['val_index_file']


    # breakpoint()
    
    # Initialize mappings
    train_mapping = {cls: [] for cls in included_classes}
    val_mapping = {cls: [] for cls in included_classes}
    
    # Process training files
    train_files = read_list_file(train_index_file)
    print("Processing training files...")
    for file_path in tqdm(train_files):
        # Check if background
        if is_background_item(file_path):
            if "background" in included_classes:
                train_mapping["background"].append(file_path)
            continue
            
        # Check if single vehicle
        is_single, vehicle = is_single_vehicle_item(file_path, vehicle_names)
        if is_single and vehicle in included_classes:
            train_mapping[vehicle].append(file_path)
    
    # Process validation files  
    val_files = read_list_file(val_index_file)
    print("Processing validation files...")
    for file_path in tqdm(val_files):
        # Check if background
        if is_background_item(file_path):
            if "background" in included_classes:
                val_mapping["background"].append(file_path)
            continue
            
        # Check if single vehicle
        is_single, vehicle = is_single_vehicle_item(file_path, vehicle_names)
        if is_single and vehicle in included_classes:
            val_mapping[vehicle].append(file_path)
    
    # Print statistics
    print("\nTraining set statistics:")
    for cls in included_classes:
        print(f"{cls}: {len(train_mapping[cls])} samples")
        
    print("\nValidation set statistics:")
    for cls in included_classes:
        print(f"{cls}: {len(val_mapping[cls])} samples")
    
    return train_mapping, val_mapping


def get_single_vehicle_items():
    with open('demo_dataset_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_index_file = config['vehicle_classification']['train_index_file']
    val_index_file = config['vehicle_classification']['val_index_file']
    vehicle_names = config['vehicle_classification']['class_names']
    max_distance_m = config['max_distance_m']
    
    train_items = read_list_file(train_index_file)
    val_items = read_list_file(val_index_file)


    
    print(f"Total train items: {len(train_items)}")
    print(f"Total val items: {len(val_items)}")
    
    # Apply max distance filter if specified
    if max_distance_m is not None:
        print(f"\nFiltering by max distance: {max_distance_m} m")
        train_items = filter_samples_by_max_distance(train_items, vehicle_names, max_distance_m)
        val_items = filter_samples_by_max_distance(val_items, vehicle_names, max_distance_m)
        print(f"After distance filter - train items: {len(train_items)}")
        print(f"After distance filter - val items: {len(val_items)}")
    
    # Filter for single-vehicle items
    print("\nFiltering for single-vehicle items...")
    train_single = filter_single_vehicle(train_items, vehicle_names)
    val_single = filter_single_vehicle(val_items, vehicle_names)
    
    print(f"Single-vehicle train items: {len(train_single)}")
    print(f"Single-vehicle val items: {len(val_single)}")
    with open('demo_dataset_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_index_file = config['vehicle_classification']['train_index_file']
    val_index_file = config['vehicle_classification']['val_index_file']
    vehicle_names = config['vehicle_classification']['class_names']
    max_distance_m = config['max_distance_m']
    
    train_items = read_list_file(train_index_file)
    val_items = read_list_file(val_index_file)
    
    print(f"Total train items: {len(train_items)}")
    print(f"Total val items: {len(val_items)}")
    
    # Apply max distance filter if specified
    if max_distance_m is not None:
        print(f"\nFiltering by max distance: {max_distance_m} m")
        train_items = filter_samples_by_max_distance(train_items, vehicle_names, max_distance_m)
        val_items = filter_samples_by_max_distance(val_items, vehicle_names, max_distance_m)
        print(f"After distance filter - train items: {len(train_items)}")
        print(f"After distance filter - val items: {len(val_items)}")
    
    # Filter for single-vehicle items
    print("\nFiltering for single-vehicle items...")
    train_single = filter_single_vehicle(train_items, vehicle_names)
    val_single = filter_single_vehicle(val_items, vehicle_names)
    
    print(f"Single-vehicle train items: {len(train_single)}")
    print(f"Single-vehicle val items: {len(val_single)}")

    return train_single, val_single