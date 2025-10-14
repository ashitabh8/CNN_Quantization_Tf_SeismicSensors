import tensorflow as tf
import numpy as np

class Args:
    """Simple args class to hold configuration"""
    def __init__(self, config):
        # print("Loading dataset config")
        self.dataset_config = config
        self.task = "vehicle_classification"


class SeismicDataset:
    """
    A TensorFlow-compatible dataset class for seismic sensor data.
    Supports lazy loading, one-hot encoding, and Welch preprocessing.
    """
    
    def __init__(self, train_mapping, val_mapping, task="vehicle_classification", 
                 spectral_processing=None, is_training=True):
        """
        Initialize the SeismicDataset.
        
        Args:
            train_mapping: Dictionary mapping vehicle names to file paths for training
            val_mapping: Dictionary mapping vehicle names to file paths for validation
            task: Task type (default: "vehicle_classification")
            spectral_processing: Configuration for spectral processing
            is_training: Whether to use training or validation data
        """
        self.task = task
        self.spectral_processing = spectral_processing or {}
        self.is_training = is_training
        
        # Choose mapping based on training/validation
        self.mapping = train_mapping if is_training else val_mapping
        
        # Create class mapping from vehicle names to indices
        self._create_class_mapping()
        
        # Flatten all file paths with their corresponding labels
        self._create_file_label_pairs()
        
        print(f"SeismicDataset initialized:")
        print(f"  Mode: {'Training' if is_training else 'Validation'}")
        print(f"  Total samples: {len(self.file_label_pairs)}")
        print(f"  Classes: {list(self.class_mapping.keys())}")
        print(f"  Class mapping: {self.class_mapping}")
    
    def _create_class_mapping(self):
        """Create mapping from vehicle names to class indices for one-hot encoding."""
        # Get unique vehicle names from both mappings
        all_vehicles = set()
        for mapping in [self.mapping]:
            all_vehicles.update(mapping.keys())
        
        # Create mapping from vehicle name to class index
        self.class_mapping = {vehicle: idx for idx, vehicle in enumerate(sorted(all_vehicles))}
        self.num_classes = len(self.class_mapping)
        
        print(f"Created class mapping: {self.class_mapping}")
        print(f"Number of classes: {self.num_classes}")
    
    def _create_file_label_pairs(self):
        """Create list of (file_path, label_index) pairs."""
        self.file_label_pairs = []
        
        for vehicle_name, file_paths in self.mapping.items():
            class_idx = self.class_mapping[vehicle_name]
            for file_path in file_paths:
                self.file_label_pairs.append((file_path, class_idx))
        
        print(f"Created {len(self.file_label_pairs)} file-label pairs")
    
    def _load_sample(self, file_path):
        """
        Load a single sample from file path.
        
        Args:
            file_path: Path to the sample file
            
        Returns:
            dict: Sample data with 'data' and 'label' keys
        """
        import torch
        
        # Load PyTorch tensor file
        sample = torch.load(file_path, weights_only=False)
        
        # Extract seismic data from the nested structure
        # Based on print_shapes function: sample['data']['shake']['seismic']
        seismic_data = sample['data']['shake']['seismic']
        
        # Convert PyTorch tensor to TensorFlow tensor
        if isinstance(seismic_data, torch.Tensor):
            seismic_data = tf.convert_to_tensor(seismic_data.numpy())
        
        return {
            'data': seismic_data,
            'label': sample.get('label', {})
        }
    
    def _reshape_data(self, data):
        """
        Reshape data from (1, 10, 20) to (1, 200).
        
        Args:
            data: TensorFlow tensor with shape (1, 10, 20)
            
        Returns:
            TensorFlow tensor with shape (1, 200)
        """
        # Reshape from (1, 10, 20) to (1, 200)
        return tf.reshape(data, (1, 200))
    
    def _apply_welch_preprocessing(self, data):
        """
        Apply Welch's method for power spectral density estimation.
        Creates a 2D spectrogram from the seismic time series data.
        
        Args:
            data: TensorFlow tensor with seismic data of shape (1, 10, 20)
            
        Returns:
            TensorFlow tensor with 2D spectrogram
        """
        if not self.spectral_processing.get('method') == 'welch':
            return data
        
        # Get Welch parameters from config
        welch_config = self.spectral_processing.get('welch', {})
        sampling_rate = self.spectral_processing.get('sampling_rate', 100.0)
        
        # Parameters for 2D spectrogram
        nperseg = welch_config.get('nperseg', 10)
        noverlap = welch_config.get('noverlap', 5)
        nfft = welch_config.get('nfft', 20)
        window_type = welch_config.get('window', 'hann')
        scaling = welch_config.get('scaling', 'density')
        detrend_type = welch_config.get('detrend', 'constant')
        
        def apply_welch_to_tensor(tensor):
            """
            Apply Welch's method to create a 2D spectrogram.
            
            Args:
                tensor: Shape (1, 10, 20) - 1 channel, 10 time steps, 20 data points per step
                       Total: 200 samples over 2 seconds at 100Hz sampling rate
                
            Returns:
                TensorFlow tensor with 2D spectrogram
            """
            # Input shape: (1, 10, 20)
            # 10 time steps × 20 samples per step = 200 total samples over 2 seconds
            # Sampling rate: 100Hz (100 samples per second)
            # We want to create a spectrogram where each time step becomes a frequency bin
            
            # Apply detrending if specified
            if detrend_type == 'constant':
                # Remove DC component (mean) from each time step
                tensor = tensor - tf.reduce_mean(tensor, axis=-1, keepdims=True)
            elif detrend_type == 'linear':
                # Remove linear trend from each time step
                tensor = tf.py_function(
                    lambda x: tf.convert_to_tensor(
                        [[tf.signal.detrend(time_step) for time_step in channel] for channel in x.numpy()], 
                        dtype=tf.float32
                    ),
                    [tensor],
                    tf.float32
                )
            
            # Apply FFT to each time step (last dimension: 20 data points)
            # Shape: (1, 10, 20) -> (1, 10, nfft//2+1)
            fft_result = tf.signal.fft(tf.cast(tensor, tf.complex64))
            
            # Take only positive frequencies
            fft_positive = fft_result[..., :nfft//2+1]
            
            # Compute power spectral density
            psd = tf.abs(fft_positive) ** 2
            
            # Apply scaling
            if scaling == 'density':
                psd = psd / sampling_rate
            
            # The result should be a 2D spectrogram: (1, 10, nfft//2+1)
            # where 10 is time dimension and nfft//2+1 is frequency dimension
            return psd
        
        return apply_welch_to_tensor(data)
    
    def _create_one_hot_label(self, class_idx):
        """
        Create one-hot encoded label from class index.
        
        Args:
            class_idx: Integer class index
            
        Returns:
            TensorFlow tensor with one-hot encoded label
        """
        return tf.one_hot(class_idx, depth=self.num_classes, dtype=tf.float32)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_label_pairs)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (data, label) where data is processed seismic data and label is one-hot encoded
        """
        file_path, class_idx = self.file_label_pairs[idx]
        
        # Load sample
        sample = self._load_sample(file_path)
        data = sample['data']
        
        # Apply Welch preprocessing if configured (creates 2D spectrogram)
        data = self._apply_welch_preprocessing(data)
        
        # Only reshape if no spectral processing was applied
        if not self.spectral_processing.get('method') == 'welch':
            # Reshape data from (1, 10, 20) to (1, 200) only if no Welch preprocessing
            data = self._reshape_data(data)
        
        # Squeeze the redundant first dimension (1) to get (10, 11) or (200,) shape
        # For Welch: (1, 10, 11) -> (10, 11)
        # For no processing: (1, 200) -> (200,)
        data = tf.squeeze(data, axis=0)
        
        # Create one-hot encoded label
        label = self._create_one_hot_label(class_idx)
        
        return data, label
    
    def to_tf_dataset(self, batch_size=32, shuffle_buffer_size=1000, num_parallel_calls=tf.data.AUTOTUNE):
        """
        Convert to TensorFlow Dataset for lazy loading and batching.
        
        Args:
            batch_size: Batch size for the dataset
            shuffle_buffer_size: Buffer size for shuffling (set to 0 to disable)
            num_parallel_calls: Number of parallel calls for data loading
            
        Returns:
            tf.data.Dataset: TensorFlow dataset ready for training
        """
        # Create dataset from indices for memory efficiency
        indices = tf.data.Dataset.from_tensor_slices(list(range(len(self))))
        
        # Shuffle indices if requested
        if shuffle_buffer_size > 0:
            indices = indices.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        # Map function to load and process data on-demand
        def load_sample(idx):
            def py_load_sample(idx_tensor):
                idx_val = idx_tensor.numpy()
                data, label = self.__getitem__(idx_val)
                return data, label
            
            # Use tf.py_function for loading PyTorch files
            return tf.py_function(
                func=py_load_sample,
                inp=[idx],
                Tout=[tf.float32, tf.float32]
            )
        
        # Apply the loading function with parallel processing
        dataset = indices.map(load_sample, num_parallel_calls=num_parallel_calls)
        
        # Set shapes explicitly (TensorFlow needs this after py_function)
        sample_data, sample_label = self.__getitem__(0)
        
        def set_shapes(data, label):
            data.set_shape(sample_data.shape)
            label.set_shape(sample_label.shape)
            return data, label
        
        dataset = dataset.map(set_shapes)
        
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Mapping from class name to count
        """
        class_counts = {}
        for _, class_idx in self.file_label_pairs:
            # Find class name from index
            class_name = None
            for name, idx in self.class_mapping.items():
                if idx == class_idx:
                    class_name = name
                    break
            
            if class_name:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts