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

# Model setup for PAN
logger.info('Initializing PAN model for parameter counting')
model = smp.PAN(
    encoder_name="resnet50",  # You can change this to other encoders like "efficientnet-b0", "densenet121", etc.
    encoder_weights="imagenet",  # Pre-trained weights
    in_channels=18,  # Your 18 input channels
    classes=2,  # Binary classification (background + target crop)
    activation=None,  # No activation for CrossEntropyLoss
    encoder_depth=5,  # Number of encoder blocks
    decoder_channels=256,  # Number of channels in decoder blocks
    decoder_use_batchnorm=True,  # Use batch normalization in decoder
    decoder_attention_type="scse",  # Use scSE attention mechanism
    decoder_use_attention=True,  # Enable attention mechanism
)

# Count and display model parameters
total_params, trainable_params = count_parameters(model)
logger.info(f'PAN Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}')
logger.info(f'Model size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)')

# Print detailed breakdown by layer
print("\n" + "="*50)
print("PAN MODEL DETAILED PARAMETER BREAKDOWN")
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

# Compare with other models
print("\n" + "="*50)
print("COMPARISON WITH OTHER MODELS")
print("="*50)

# UPerNet for comparison
logger.info('Initializing UPerNet model for comparison')
upernet_model = smp.UPerNet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=18,
    classes=2,
    activation=None,
    encoder_depth=5,
    psp_channels=512,
    psp_use_batchnorm=True,
    psp_dropout=0.2,
    decoder_channels=256,
    decoder_use_batchnorm=True,
    decoder_attention_type=None,
    decoder_use_attention=False,
)

upernet_total, upernet_trainable = count_parameters(upernet_model)
print(f"UPerNet: {upernet_total:,} parameters ({upernet_total * 4 / (1024**2):.2f} MB)")

# UNet for comparison
logger.info('Initializing UNet model for comparison')
unet_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=18,
    classes=2,
    activation=None,
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    decoder_attention_type=None,
)

unet_total, unet_trainable = count_parameters(unet_model)
print(f"UNet: {unet_total:,} parameters ({unet_total * 4 / (1024**2):.2f} MB)")

# FPN for comparison
logger.info('Initializing FPN model for comparison')
fpn_model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=18,
    classes=2,
    activation=None,
    encoder_depth=5,
    pyramid_channels=256,
    segmentation_channels=128,
    dropout=0.2,
    upsampling=4,
)

fpn_total, fpn_trainable = count_parameters(fpn_model)
print(f"FPN: {fpn_total:,} parameters ({fpn_total * 4 / (1024**2):.2f} MB)")

print(f"\nPAN: {total_params:,} parameters ({total_params * 4 / (1024**2):.2f} MB)")
print("="*50)
