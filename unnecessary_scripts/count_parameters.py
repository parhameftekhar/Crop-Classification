import torch
import segmentation_models_pytorch as smp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to count model parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Model setup (same as in train_UPerNet.py)
logger.info('Initializing UPerNet model for parameter counting')
model = smp.UPerNet(
    encoder_name="resnet50",  # You can change this to other encoders like "efficientnet-b0", "densenet121", etc.
    encoder_weights="imagenet",  # Pre-trained weights
    in_channels=18,  # Your 18 input channels
    classes=2,  # Binary classification (background + target crop)
    activation=None,  # No activation for CrossEntropyLoss
    encoder_depth=5,  # Number of encoder blocks
    psp_channels=512,  # Number of channels in PSP module
    psp_use_batchnorm=True,  # Use batch normalization in PSP
    psp_dropout=0.2,  # Dropout rate in PSP
    decoder_channels=256,  # Number of channels in decoder blocks (single integer, not tuple)
    decoder_use_batchnorm=True,  # Use batch normalization in decoder
    decoder_attention_type=None,  # No attention in decoder
    decoder_use_attention=False,  # Disable attention mechanism
)

# Count and display model parameters
total_params, trainable_params = count_parameters(model)
logger.info(f'Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}')
logger.info(f'Model size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)')

# Print detailed breakdown by layer
print("\n" + "="*50)
print("DETAILED PARAMETER BREAKDOWN")
print("="*50)
total_count = 0
for name, param in model.named_parameters():
    param_count = param.numel()
    total_count += param_count
    print(f"{name}: {param_count:,} parameters")
    if param.requires_grad:
        print(f"  -> Trainable")
    else:
        print(f"  -> Frozen")

print(f"\nTotal parameters: {total_count:,}")
print(f"Model size: {total_count * 4 / (1024**2):.2f} MB (assuming float32)")
print(f"Model size: {total_count * 4 / (1024**3):.2f} GB (assuming float32)")
