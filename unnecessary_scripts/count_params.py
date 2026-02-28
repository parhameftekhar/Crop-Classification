import torch
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3, fcn_resnet50, FCN_ResNet50_Weights
from model import FeatureExtractor

def count_model_params(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024*1024)  # Assuming float32 (4 bytes per parameter)
    
    print(f"\n{model_name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (MB): {model_size_mb:.2f}")
    
    return total_params, trainable_params, model_size_mb

# MANet model (exact same as in train_MAnet.py)
print("=== Parameter Count Comparison ===")
manet = smp.MAnet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=18,
    classes=2,
    activation=None,
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    decoder_pab_channels=64,
    decoder_use_attention=True,
)
manet_params = count_model_params(manet, "MANet")

# UNet model (exact same as in train_Unet.py)
unet = smp.Unet(
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
unet_params = count_model_params(unet, "UNet")

# FPN model (exact same as in train_FPN.py)
fpn = smp.FPN(
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
fpn_params = count_model_params(fpn, "FPN")

# FCN model (exact same as in train_fcn_binary.py)
fcn_weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
fcn_model = fcn_resnet50(num_classes=2)

# Modify the first convolution layer for 18 input channels (same as in training script)
original_conv = fcn_model.backbone.conv1
new_conv = torch.nn.Conv2d(
    in_channels=18,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None,
)
fcn_model.backbone.conv1 = new_conv
fcn_params = count_model_params(fcn_model, "FCN")

# DeepLabV3 model (exact same as in train_deeplabv3_binary.py)
weights = deeplabv3.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
deeplabv3_model = deeplabv3_resnet50(num_classes=2)

# Modify the first convolution layer for 18 input channels (same as in training script)
original_conv = deeplabv3_model.backbone.conv1
new_conv = torch.nn.Conv2d(
    in_channels=18,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None,
)
deeplabv3_model.backbone.conv1 = new_conv
deeplabv3_params = count_model_params(deeplabv3_model, "DeepLabV3")

# Segformer model (exact same as in train_segformer.py)
try:
    config = SegformerConfig(
        num_labels=5,
        image_size=224,
        num_channels=18,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        num_attention_heads=[1, 2, 5, 8],
        drop_path_rate=0.1,
        semantic_loss_ignore_index=255,
        loss_type="weighted_ce",
        label_smoothing=0.1
    )
    segformer = SegformerForSemanticSegmentation(config)
    
    # Modify the first conv layer to accept 18 input channels (same as in training script)
    old_proj = segformer.segformer.encoder.patch_embeddings[0].proj
    new_proj = torch.nn.Conv2d(
        in_channels=18,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    segformer.segformer.encoder.patch_embeddings[0].proj = new_proj
    
    segformer_params = count_model_params(segformer, "Segformer")
except ImportError:
    print("\nSegformer: transformers library not available")
    segformer_params = None

# FeatureExtractor model (exact same as in train_contrastive_learning.py)
feature_extractor = FeatureExtractor(
    num_block=4,
    kernel_size=9,
    stride=1,
    padding=4,
    num_channel_in=18,
    num_channel_internal=36,
    num_channel_out=18,
    matrix_size=18,
    device='cpu'  # Use CPU for parameter counting
)
feature_extractor_params = count_model_params(feature_extractor, "FeatureExtractor")

print("\n=== Summary ===")
models = [
    ("MANet", manet_params),
    ("UNet", unet_params),
    ("FPN", fpn_params),
    ("FCN", fcn_params),
    ("DeepLabV3", deeplabv3_params),
    ("FeatureExtractor", feature_extractor_params),
]

if segformer_params:
    models.append(("Segformer", segformer_params))

# Sort by parameter count
models.sort(key=lambda x: x[1][0])
for name, (total, trainable, size) in models:
    print(f"{name:12} - {total:>10,} params ({size:>6.1f} MB)")
