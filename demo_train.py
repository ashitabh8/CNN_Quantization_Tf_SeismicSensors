from re import I
from dataset import read_list_file, filter_samples_by_max_distance, filter_single_vehicle, MultiModalDataset, Args
import yaml


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

if __name__ == "__main__":
    train_single, val_single = get_single_vehicle_items()
    config_path = 'demo_dataset_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        args = Args(config)
    train_dataset = MultiModalDataset(args, train_single)
    # val_dataset = MultiModalDataset(args, val_single)

    print(f"Train single: {train_single[:10]}")
    print(f"Val single: {val_single[:10]}")