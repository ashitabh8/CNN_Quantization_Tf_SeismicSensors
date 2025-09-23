#!/usr/bin/env python3
"""
Test script to verify the ResNet-style CNN model fits within 700KB target
"""

import tensorflow as tf
from models import create_resnet_style_cnn, print_resnet_model_info

def test_resnet_model():
    """Test the ResNet model and verify memory usage"""
    print("=" * 80)
    print("TESTING RESNET-STYLE CNN MODEL")
    print("=" * 80)
    
    # Test with seismic data input shape
    input_shape = (2, 7, 256)
    num_classes = 10
    target_memory_kb = 700
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Target memory: {target_memory_kb} KB")
    print("-" * 80)
    
    # Create the ResNet model
    model = create_resnet_style_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        target_memory_kb=target_memory_kb
    )
    
    # Build the model
    model.build(input_shape=(None,) + input_shape)
    
    # Print detailed model information
    memory_kb = print_resnet_model_info(model, target_memory_kb)
    
    # Test model compilation
    print("\n" + "=" * 80)
    print("TESTING MODEL COMPILATION")
    print("=" * 80)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model compiled successfully!")
    
    # Test forward pass with batch
    print("\n" + "=" * 80)
    print("TESTING FORWARD PASS")
    print("=" * 80)
    
    batch_size = 8
    test_input = tf.random.normal((batch_size,) + input_shape)
    print(f"Test input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    # Verify output probabilities sum to 1
    output_sums = tf.reduce_sum(output, axis=1)
    print(f"Output probability sums (should be ~1.0): {output_sums.numpy()}")
    
    print("\n" + "=" * 80)
    print("RESNET MODEL TEST RESULTS")
    print("=" * 80)
    print(f"Memory usage: {memory_kb:.2f} KB / {target_memory_kb} KB")
    print(f"Memory efficiency: {'✅ PASS' if memory_kb < target_memory_kb else '❌ FAIL'}")
    print(f"Model compilation: ✅ PASS")
    print(f"Forward pass: ✅ PASS")
    print("=" * 80)
    
    return model, memory_kb

if __name__ == "__main__":
    model, memory_usage = test_resnet_model()
