# test_pspnet_dimensions.py

import torch
import segmentation_models_pytorch as smp
import numpy as np

def test_pspnet_dimensions():
    print("Testing PSPNet model dimensions...")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create PSPNet model with same configuration as training script
    model = smp.PSPNet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=18,
        classes=2,
        activation=None,
        encoder_depth=3,
        psp_use_batchnorm=True,
        psp_dropout=0.2,
        psp_conv_filters=256,
        aux_params=None,
        upsampling=8
    )
    
    # Move model to GPU
    model = model.to(device)
    
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different input sizes
    test_sizes = [(224, 224), (256, 256), (512, 512)]
    
    for height, width in test_sizes:
        print(f"\nTesting with input size: {height}x{width}")
        
        # Create random input tensor (batch_size=2, channels=18, height, width)
        input_tensor = torch.randn(2, 18, height, width).to(device)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output tensor shape: {output.shape}")
        
        # Check if output matches input spatial dimensions
        expected_height = height
        expected_width = width
        actual_height = output.shape[2]
        actual_width = output.shape[3]
        
        if actual_height == expected_height and actual_width == expected_width:
            print(f"✅ PASS: Output dimensions match input ({actual_height}x{actual_width})")
        else:
            print(f"❌ FAIL: Expected {expected_height}x{expected_width}, got {actual_height}x{actual_width}")
    
    # Test with batch size 1 (like in training)
    print(f"\nTesting with batch size 1:")
    input_tensor = torch.randn(1, 18, 224, 224).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output tensor shape: {output.shape}")
    
    if output.shape[2:] == (224, 224):
        print(f"✅ PASS: Output dimensions correct for training")
    else:
        print(f"❌ FAIL: Output dimensions incorrect for training")
    
    # Test with multiple batches (like in training)
    print(f"\nTesting with batch size 16 (training batch size):")
    input_tensor = torch.randn(16, 18, 224, 224).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output tensor shape: {output.shape}")
    
    if output.shape == (16, 2, 224, 224):
        print(f"✅ PASS: Output dimensions correct for training batches")
    else:
        print(f"❌ FAIL: Expected (16, 2, 224, 224), got {output.shape}")
    
    print(f"\nTest completed!")

if __name__ == "__main__":
    test_pspnet_dimensions()
