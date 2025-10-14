from re import I
import yaml
import tensorflow as tf
from dataset import Args, SeismicDataset
from dataset_utils import create_mapping_vehicle_name_to_file_path, filter_samples_by_max_distance, give_count_from_mapping, print_shapes


def setup_dataset(config):

    train_mapping, val_mapping = create_mapping_vehicle_name_to_file_path(config)
    train_mapping = filter_samples_by_max_distance(train_mapping, config['max_distance_m'])
    val_mapping = filter_samples_by_max_distance(val_mapping, config['max_distance_m'])
    batch_size = config['batch_size']

    give_count_from_mapping(train_mapping)
    give_count_from_mapping(val_mapping)
    print_shapes(train_mapping)
    print_shapes(val_mapping)

    train_dataset = SeismicDataset(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        task=args.task,
        spectral_processing=config.get('spectral_processing', {}),
        is_training=True
    )

    val_dataset = SeismicDataset(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        task=args.task,
        spectral_processing=config.get('spectral_processing', {}),
        is_training=False
    )

    train_tf_dataset = train_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_tf_dataset = val_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle_buffer_size=0,  # No shuffling for validation
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return train_tf_dataset, val_tf_dataset

if __name__ == "__main__":
    config_path = 'demo_dataset_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        args = Args(config)

    # print(args.dataset_config)
    train_tf_dataset, val_tf_dataset = setup_dataset(config)
    # Test a sample batch
    print("\n=== Testing sample batch ===")
    for batch_data, batch_labels in train_tf_dataset.take(1):
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Sample labels: {batch_labels.numpy()[:3]}")  # Show first 3 labels
        break