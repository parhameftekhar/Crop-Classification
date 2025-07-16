# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Script to process raster files, split them into patches, and create train/val/test sets.
# """

# import os
# import argparse
# from pathlib import Path
# from data_manager import process_all_files


# def main():
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Process raster files into patches and split into train/val/test sets.')
    
#     parser.add_argument('--input_dir', type=str, required=True,
#                         help='Directory containing raster files')
    
#     parser.add_argument('--output_dir', type=str, required=True,
#                         help='Directory to save processed data')
    
#     parser.add_argument('--patch_height', type=int, default=64,
#                         help='Height of each patch (default: 64)')
    
#     parser.add_argument('--patch_width', type=int, default=64,
#                         help='Width of each patch (default: 64)')
    
#     parser.add_argument('--train_ratio', type=float, default=0.7,
#                         help='Proportion of data to use for training (default: 0.7)')
    
#     parser.add_argument('--val_ratio', type=float, default=0.15,
#                         help='Proportion of data to use for validation (default: 0.15)')
    
#     parser.add_argument('--test_ratio', type=float, default=0.15,
#                         help='Proportion of data to use for testing (default: 0.15)')
    
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed for reproducibility (default: 42)')
    
#     args = parser.parse_args()
    
#     # Ensure input directory exists
#     input_dir = Path(args.input_dir)
#     if not input_dir.exists():
#         print(f"Error: Input directory '{input_dir}' does not exist.")
#         return
    
#     # Create output directory if it doesn't exist
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Print parameters
#     print("Processing parameters:")
#     print(f"  Input directory: {input_dir}")
#     print(f"  Output directory: {output_dir}")
#     print(f"  Patch size: {args.patch_height}x{args.patch_width}")
#     print(f"  Train/Val/Test ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
#     print(f"  Random seed: {args.seed}")
    
#     # Run the processing function
#     print("\nStarting data processing...")
#     train_patches, val_patches, test_patches = process_all_files(
#         input_dir=input_dir,
#         output_dir=output_dir,
#         h=args.patch_height,
#         w=args.patch_width,
#         train_ratio=args.train_ratio,
#         val_ratio=args.val_ratio,
#         test_ratio=args.test_ratio,
#         seed=args.seed
#     )
    
#     print("\nData processing completed successfully!")
#     print(f"  Train patches: {train_patches.shape}")
#     print(f"  Validation patches: {val_patches.shape}")
#     print(f"  Test patches: {test_patches.shape}")
#     print(f"\nProcessed data saved to: {output_dir}")


# if __name__ == "__main__":
#     main() 



# from utils import analyze_data_splits

# # Analyze the data splits
# results = analyze_data_splits("./training_data")