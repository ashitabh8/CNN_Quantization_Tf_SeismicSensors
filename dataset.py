import os
import tensorflow as tf
import numpy as np
from random import shuffle
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

"""
TensorFlow Keras Dataset Class for Seismic Sensor Data

USAGE EXAMPLES:

1. Basic Usage:
   ```python
   import yaml
   
   # Load dataset configuration
   with open('dataset_config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Create args object (or use your existing args)
   class Args:
       def __init__(self):
           self.dataset_config = config
           self.task = "vehicle_classification"
   
   args = Args()
   
   # Create dataset instances
   train_dataset = MultiModalDataset(
       args=args,
       index_file=config['vehicle_classification']['train_index_file']
   )
   val_dataset = MultiModalDataset(
       args=args, 
       index_file=config['vehicle_classification']['val_index_file']
   )
   ```

2. Convert to TensorFlow Dataset for Keras Training:
   ```python
   # Standard conversion (memory-efficient with parallel loading)
   train_tf_dataset = train_dataset.to_tf_dataset(
       batch_size=32, 
       shuffle_buffer_size=1000,
       num_parallel_calls=4  # Parallel data loading
   )
   val_tf_dataset = val_dataset.to_tf_dataset(batch_size=32, shuffle_buffer_size=0)
   
   # For very large datasets, use lazy loading with caching
   train_tf_dataset = train_dataset.to_tf_dataset_lazy(
       batch_size=32,
       shuffle_buffer_size=1000,
       cache_size=100  # Keep 100 samples in memory cache
   )
   
   # Use with Keras model
   model = tf.keras.Sequential([
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')  # 10 classes from config
   ])
   
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   
   # Train the model
   history = model.fit(
       train_tf_dataset,
       validation_data=val_tf_dataset,
       epochs=10
   )
   ```

3. Access Individual Samples:
   ```python
   # Get a single sample
   data, label, idx = train_dataset[0]
   print(f"Data shape: {data.shape}")
   print(f"Label: {label}")
   print(f"Sample index: {idx}")
   
   # Iterate through dataset
   for i in range(len(train_dataset)):
       data, label, idx = train_dataset[i]
       # Process data...
   ```

4. Balanced Sampling:
   ```python
   # Create dataset with balanced sampling
   balanced_dataset = MultiModalDataset(
       args=args,
       index_file=config['vehicle_classification']['train_index_file'],
       balanced_sample=True
   )
   ```

5. Partial Dataset (for debugging/testing):
   ```python
   # Use only 10% of the data
   small_dataset = MultiModalDataset(
       args=args,
       index_file=config['vehicle_classification']['train_index_file'],
       label_ratio=0.1
   )
   ```

6. Memory-Efficient Options for Large Datasets:
   ```python
   # Option 1: Parallel loading with controlled memory usage
   large_dataset = train_dataset.to_tf_dataset(
       batch_size=16,  # Smaller batches for large data
       shuffle_buffer_size=500,  # Smaller shuffle buffer
       num_parallel_calls=2  # Limit parallel workers
   )
   
   # Option 2: Lazy loading with LRU cache
   large_dataset = train_dataset.to_tf_dataset_lazy(
       batch_size=16,
       shuffle_buffer_size=500,
       cache_size=50  # Cache only 50 most recent samples
   )
   
   # Option 3: No caching for maximum memory efficiency
   large_dataset = train_dataset.to_tf_dataset_lazy(
       batch_size=16,
       shuffle_buffer_size=500,
       cache_size=0  # No caching
   )
   ```
"""


class MultiModalDataset:
    def __init__(self, args, index_file, label_ratio=1, balanced_sample=False):
        """
        Args:
            modalities (_type_): The list of modalities
            classes (_type_): The list of classes
            index_file (_type_): The list of sample file names
            sample_path (_type_): The base sample path.

        Sample:
            - label: Tensor
            - flag
                - phone
                    - audio: True
                    - acc: False
            - data:
                -phone
                    - audio: Tensor
                    - acc: Tensor
        """
        self.args = args
        self.sample_files = list(np.loadtxt(index_file, dtype=str))
        
        # for i in range(len(self.sample_files)):
        #     if not os.path.exists(self.sample_files[i]):
        #         self.sample_files[i] = self.sample_files[i].replace("/home/sl29/", "/data/sl29/home_sl29/")

        # Apply class filtering if specified in config
        self._apply_class_filtering()
        
        # Create class mapping for one-hot encoding
        self._create_class_mapping()

        if balanced_sample:
            self.load_sample_labels()

    def _extract_class_from_sample(self, filepath):
        """
        Extract class from the actual sample data by loading the file.
        
        Args:
            filepath (str): Full path to the sample file
            
        Returns:
            int: Class index extracted from sample label, or None if extraction fails
        """
        try:
            import torch
            
            # Load the sample file
            sample = torch.load(filepath, weights_only=True)
            
            # Extract label based on task type (same logic as in __getitem__)
            if isinstance(sample["label"], dict):
                if self.args.task == "vehicle_classification":
                    label = sample["label"]["vehicle_type"]
                elif self.args.task == "distance_classification":
                    label = sample["label"]["distance"]
                elif self.args.task == "speed_classification":
                    label = sample["label"]["speed"] // 5 - 1
                elif self.args.task == "terrain_classification":
                    label = sample["label"]["terrain"]
                else:
                    print(f"Warning: Unknown task {self.args.task} for file {filepath}")
                    return None
            else:
                label = sample["label"]
            
            # Handle tensor values
            if hasattr(label, 'numpy'):
                return int(label.numpy())
            elif hasattr(label, 'item'):
                return int(label.item())
            else:
                return int(label)
                
        except Exception as e:
            print(f"Warning: Could not load sample from {filepath}: {e}")
            return None

    def _apply_class_filtering(self):
        """
        Filter sample files based on included_classes configuration.
        Also prints statistics about class distribution.
        """
        # Get included classes from config
        task_config = self.args.dataset_config.get(self.args.task, {})
        included_classes = task_config.get('included_classes', None)
        
        if included_classes is None or len(included_classes) == 0:
            print("No class filtering specified - using all classes")
            self._print_class_statistics()
            return
        
        print(f"Applying class filtering for task: {self.args.task}")
        print(f"Included classes: {included_classes}")
        
        # Filter samples based on class
        filtered_files = []
        class_counts = {}
        
        print("Loading samples to extract actual labels...")
        if tqdm is not None:
            iterator = tqdm(self.sample_files, desc="Extracting labels", unit="file")
        else:
            print("(Install tqdm for a progress bar: pip install tqdm)")
            iterator = self.sample_files
        for filepath in iterator:
            class_idx = self._extract_class_from_sample(filepath)
            if class_idx is not None and class_idx in included_classes:
                filtered_files.append(filepath)
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
            elif class_idx is None:
                print(f"Warning: Could not extract class from sample: {filepath}")
        
        # Update sample files
        original_count = len(self.sample_files)
        self.sample_files = filtered_files
        filtered_count = len(self.sample_files)
        
        print(f"Class filtering results:")
        print(f"  Original samples: {original_count}")
        print(f"  Filtered samples: {filtered_count}")
        print(f"  Removed samples: {original_count - filtered_count}")
        
        # Print class distribution
        print(f"\nClass distribution after filtering:")
        for class_idx in sorted(class_counts.keys()):
            class_name = task_config.get('class_names', ['unknown'])[class_idx] if class_idx < len(task_config.get('class_names', [])) else f"class_{class_idx}"
            print(f"  Class {class_idx} ({class_name}): {class_counts[class_idx]} samples")

    def _print_class_statistics(self):
        """
        Print statistics about class distribution in the dataset.
        """
        task_config = self.args.dataset_config.get(self.args.task, {})
        class_names = task_config.get('class_names', [])
        
        class_counts = {}
        print("Loading samples to extract actual labels for statistics...")
        for i, filepath in enumerate(self.sample_files):
            if i % 100 == 0:  # Progress indicator
                print(f"  Processed {i}/{len(self.sample_files)} files...")
            
            class_idx = self._extract_class_from_sample(filepath)
            if class_idx is not None:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        print(f"\nClass distribution (all classes):")
        for class_idx in sorted(class_counts.keys()):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            print(f"  Class {class_idx} ({class_name}): {class_counts[class_idx]} samples")

    def _create_class_mapping(self):
        """
        Create mapping from original class indices to sequential indices for one-hot encoding.
        Maps included_classes [0,1,8] to [0,1,2] for proper one-hot encoding.
        """
        task_config = self.args.dataset_config.get(self.args.task, {})
        included_classes = task_config.get('included_classes', None)
        
        if included_classes is None or len(included_classes) == 0:
            # If no filtering, use all classes sequentially
            self.class_mapping = {i: i for i in range(task_config.get('num_classes', 10))}
            self.num_classes_for_training = task_config.get('num_classes', 10)
            print("No class filtering - using all classes sequentially")
        else:
            # Create mapping from original indices to sequential indices
            self.class_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(sorted(included_classes))}
            self.num_classes_for_training = len(included_classes)
            print(f"Created class mapping: {self.class_mapping}")
            print(f"Number of classes for training: {self.num_classes_for_training}")
        
        # Print mapping for verification
        print("Class mapping (original -> new):")
        for original, new in sorted(self.class_mapping.items()):
            class_name = task_config.get('class_names', ['unknown'])[original] if original < len(task_config.get('class_names', [])) else f"class_{original}"
            print(f"  {original} ({class_name}) -> {new}")

    def load_sample_labels(self):
        sample_labels = []
        label_count = [0 for i in range(self.num_classes_for_training)]

        for idx in range(len(self.sample_files)):
            _, label, _ = self.__getitem__(idx)
            if isinstance(label, tf.Tensor):
                # For one-hot encoded labels, get the class index
                label = tf.argmax(label).numpy() if len(label.shape) > 0 and label.shape[0] > 1 else label.numpy()
            sample_labels.append(int(label))
            label_count[int(label)] += 1

        self.sample_weights = []
        self.epoch_len = int(np.max(label_count) * len(label_count))
        for sample_label in sample_labels:
            self.sample_weights.append(1 / label_count[sample_label])

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # Load PyTorch tensor file and convert to TensorFlow
        import torch
        sample = torch.load(self.sample_files[idx], weights_only=True)
        
        # Convert PyTorch tensor data to TensorFlow tensor
        data = sample["data"]
        if isinstance(data, torch.Tensor):
            data = tf.convert_to_tensor(data.numpy())
        elif isinstance(data, dict):
            # Handle nested dictionary structure
            data = self._convert_dict_tensors(data)

        # ACIDS and Parkland
        if isinstance(sample["label"], dict):
            if self.args.task == "vehicle_classification":
                label = sample["label"]["vehicle_type"]
            elif self.args.task == "distance_classification":
                label = sample["label"]["distance"]
            elif self.args.task == "speed_classification":
                label = sample["label"]["speed"] // 5 - 1
            elif self.args.task == "terrain_classification":
                label = sample["label"]["terrain"]
            else:
                raise ValueError(f"Unknown task: {self.args.task}")
        else:
            label = sample["label"]

        # Convert label to TensorFlow tensor
        if isinstance(label, torch.Tensor):
            label = tf.convert_to_tensor(label.numpy())
        else:
            label = tf.convert_to_tensor(label)
        
        # Apply class mapping and convert to one-hot encoding
        label = self._apply_class_mapping_and_onehot(label)
        
        # Apply spectral processing based on configuration
        if hasattr(self.args, 'spectral_processing'):
            spectral_config = self.args.spectral_processing
            method = spectral_config.get('method', 'none')
            
            if method == 'welch':
                print("Applying Welch's method")
                data = self._apply_welch(data)
            elif method == 'fft':
                data = self._apply_fft(data)
            # If method is 'none' or not specified, no spectral processing is applied
        
        # Legacy support for fft_processing config
        elif hasattr(self.args, 'fft_processing') and self.args.fft_processing.get('enabled', False):
            data = self._apply_fft(data)
        
        return data, label, idx
    
    def _apply_class_mapping_and_onehot(self, label):
        """
        Apply class mapping and convert to one-hot encoding.
        
        Args:
            label: TensorFlow tensor with original class index
            
        Returns:
            TensorFlow tensor with one-hot encoded label
        """
        # Convert to numpy for easier manipulation
        if hasattr(label, 'numpy'):
            original_class = int(label.numpy())
        else:
            original_class = int(label)
        
        # Apply class mapping
        if original_class in self.class_mapping:
            mapped_class = self.class_mapping[original_class]
        else:
            print(f"Warning: Class {original_class} not found in mapping. Using 0 as default.")
            mapped_class = 0
        
        # Convert to one-hot encoding
        one_hot = tf.one_hot(mapped_class, depth=self.num_classes_for_training, dtype=tf.float32)
        
        return one_hot
    
    def _convert_dict_tensors(self, data_dict):
        """Recursively convert PyTorch tensors in nested dictionaries to TensorFlow tensors"""
        import torch
        converted = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                converted[key] = tf.convert_to_tensor(value.numpy())
            elif isinstance(value, dict):
                converted[key] = self._convert_dict_tensors(value)
            else:
                converted[key] = value
        return converted
    
    def _apply_fft(self, data):
        """
        Apply FFT to time series data while maintaining dimensions.
        
        Args:
            data: TensorFlow tensor with shape (2, 7, 256) or nested dict containing such tensors
            
        Returns:
            TensorFlow tensor in frequency domain with same shape as input
        """
        # Apply FFT to the last dimension (256 sample points)
        # Input shape: (2, 7, 256) -> Output shape: (2, 7, 256)
        fft_data = tf.signal.fft(tf.cast(data, tf.complex64))
        
        # Take the magnitude of the FFT to get real values
        # This gives us the frequency domain representation
        fft_magnitude = tf.abs(fft_data)
        
        # Convert back to float32 for consistency
        return tf.cast(fft_magnitude, tf.float32)
    
    def _apply_welch(self, data):
        """
        Apply Welch's method for power spectral density estimation to time series data.
        This is more robust than FFT as it reduces noise through averaging.
        
        Args:
            data: TensorFlow tensor with shape (2, 7, 256) or nested dict containing such tensors
            
        Returns:
            TensorFlow tensor with power spectral density estimates
        """
        # Get Welch parameters from config
        spectral_config = getattr(self.args, 'spectral_processing', {})
        welch_config = spectral_config.get('welch', {})
        sampling_rate = spectral_config.get('sampling_rate', 512.0)
        
        # Default parameters - configured for 8 Hz power bin sizes
        # to accommodate speed-related shifts in engine-driven features
        nperseg = welch_config.get('nperseg', 32)
        noverlap = welch_config.get('noverlap', 16)
        nfft = welch_config.get('nfft', 64)  # 512 Hz / 64 = 8 Hz bins
        window_type = welch_config.get('window', 'hann')
        scaling = welch_config.get('scaling', 'density')
        detrend_type = welch_config.get('detrend', False)
        
        def apply_welch_to_tensor(tensor):
            """
            Apply Welch's method to pre-windowed data.
            
            Args:
                tensor: TensorFlow tensor with shape (2, 7, 256) where:
                       - 2: high/low channels
                       - 7: pre-windowed segments with half overlap
                       - 256: sample points per segment
                
            Returns:
                TensorFlow tensor with PSD estimates of shape (2, nfft//2+1)
            """
            # Input shape: (2, 7, 256)
            # We want to apply FFT to each of the 7 windows and average them
            
            # Apply detrending if specified
            if detrend_type == 'constant':
                # Remove DC component (mean) from each window
                tensor = tensor - tf.reduce_mean(tensor, axis=-1, keepdims=True)
            elif detrend_type == 'linear':
                # Remove linear trend from each window
                tensor = tf.py_function(
                    lambda x: tf.convert_to_tensor(
                        [[tf.signal.detrend(window) for window in channel] for channel in x.numpy()], 
                        dtype=tf.float32
                    ),
                    [tensor],
                    tf.float32
                )
            # If detrend_type is False or None, no detrending is applied
            
            # Apply FFT to each window (last dimension: 256 samples)
            # Shape: (2, 7, 256) -> (2, 7, nfft//2+1)
            fft_result = tf.signal.fft(tf.cast(tensor, tf.complex64))
            
            # Take only the positive frequencies (first nfft//2+1 components)
            # This gives us the one-sided spectrum
            fft_positive = fft_result[..., :nfft//2+1]
            
            # Compute power spectral density
            psd = tf.abs(fft_positive) ** 2
            
            # Average across the 7 windows (Welch's method characteristic)
            # Shape: (2, 7, nfft//2+1) -> (2, nfft//2+1)
            psd_avg = tf.reduce_mean(psd, axis=1)
            
            # Apply scaling
            if scaling == 'density':
                # Convert to power spectral density
                # PSD = Power / (sampling_rate * window_energy)
                # For pre-windowed data, we assume unit window energy
                psd_avg = psd_avg / sampling_rate
            
            return psd_avg
        
        # Handle nested dictionary structure
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, tf.Tensor):
                    result[key] = apply_welch_to_tensor(value)
                else:
                    result[key] = value
            return result
        else:
            return apply_welch_to_tensor(data)
    
    def to_tf_dataset(self, batch_size=32, shuffle_buffer_size=1000, num_parallel_calls=tf.data.AUTOTUNE):
        """
        Convert to TensorFlow Dataset for use with Keras models.
        Memory-efficient implementation that loads and converts data on-demand.
        
        Args:
            batch_size (int): Batch size for the dataset
            shuffle_buffer_size (int): Buffer size for shuffling (set to 0 to disable)
            num_parallel_calls (int): Number of parallel calls for data loading
            
        Returns:
            tf.data.Dataset: TensorFlow dataset ready for training
        """
        # Create dataset from file indices for memory efficiency
        indices = tf.data.Dataset.from_tensor_slices(list(range(len(self))))
        
        # Shuffle indices if requested
        if shuffle_buffer_size > 0:
            indices = indices.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        # Map function to load and convert data on-demand
        def load_sample(idx):
            def py_load_sample(idx_tensor):
                idx_val = idx_tensor.numpy()
                data, label, _ = self.__getitem__(idx_val)
                return data, label
            
            # Use tf.py_function for loading PyTorch files
            return tf.py_function(
                func=py_load_sample,
                inp=[idx],
                Tout=[tf.float32, tf.float32]  # Changed to float32 for one-hot encoded labels
            )
        
        # Apply the loading function with parallel processing
        dataset = indices.map(load_sample, num_parallel_calls=num_parallel_calls)
        
        # Set shapes explicitly (TensorFlow needs this after py_function)
        sample_data, sample_label, _ = self.__getitem__(0)
        if isinstance(sample_data, dict):
            # Handle nested dictionary case
            def set_shapes(data, label):
                for key, value in sample_data.items():
                    if key in data:
                        data[key].set_shape(value.shape)
                label.set_shape(sample_label.shape)
                return data, label
            dataset = dataset.map(set_shapes)
        else:
            def set_shapes(data, label):
                data.set_shape(sample_data.shape)
                label.set_shape(sample_label.shape)
                return data, label
            dataset = dataset.map(set_shapes)
        
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def to_tf_dataset_lazy(self, batch_size=32, shuffle_buffer_size=1000, cache_size=100):
        """
        Alternative memory-efficient implementation with optional caching.
        
        Args:
            batch_size (int): Batch size for the dataset
            shuffle_buffer_size (int): Buffer size for shuffling
            cache_size (int): Number of samples to keep in memory cache (0 to disable)
            
        Returns:
            tf.data.Dataset: TensorFlow dataset with lazy loading
        """
        from functools import lru_cache
        
        # Optional LRU cache for frequently accessed samples
        if cache_size > 0:
            @lru_cache(maxsize=cache_size)
            def cached_getitem(idx):
                return self.__getitem__(idx)
            get_item_func = cached_getitem
        else:
            get_item_func = self.__getitem__
        
        def generator():
            indices = list(range(len(self)))
            if shuffle_buffer_size > 0:
                from random import shuffle
                shuffle(indices)
            
            for idx in indices:
                data, label, _ = get_item_func(idx)
                # print(f"Label: {label}")  # Commented out to reduce output
                yield data, label
            return
        
        # Get sample data to determine output signature
        sample_data, sample_label, _ = self.__getitem__(0)
        # print(f"Sample label: {sample_label}")
        # Create output signature
        def create_tensor_spec(data):
            """Recursively create TensorSpec for nested dictionary structures"""
            if isinstance(data, dict):
                spec = {}
                for key, value in data.items():
                    spec[key] = create_tensor_spec(value)
                return spec
            else:
                return tf.TensorSpec(shape=data.shape, dtype=data.dtype)
        
        data_spec = create_tensor_spec(sample_data)
        
        label_spec = tf.TensorSpec(shape=sample_label.shape, dtype=sample_label.dtype)
        
        # Create dataset with generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(data_spec, label_spec)
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset