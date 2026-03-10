#!/bin/bash

# run_binary_benchmark.sh
# Script to run binary training benchmarks sequentially with modified ratios (0 to 1)
# Run this from the project root directory

echo "Starting binary benchmark training sequence..."
echo "Note: Ncut is skipped as requested."

# List of scripts to run as modules (dot notation, no .py extension)
MODULES=(
    "benchmark_scripts.binary_case.train_deeplabv3_binary"
    "benchmark_scripts.binary_case.train_FPN"
    "benchmark_scripts.binary_case.train_MAnet"
    "benchmark_scripts.binary_case.train_PAN"
    "benchmark_scripts.binary_case.train_UPerNet"
    "benchmark_scripts.binary_case.train_Unet"
    "benchmark_scripts.binary_case.train_fcn_binary"
    "benchmark_scripts.binary_case.train_pspnet"
    "benchmark_scripts.binary_case.train_segformer"
    "benchmark_scripts.binary_case.train_segmenter"
    "benchmark_scripts.binary_case.train_shallowcnn_binary"
)

# Create logs directory if it doesn't exist
mkdir -p logs/benchmark/binary_case

for module in "${MODULES[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running module: $module..."
    echo "----------------------------------------------------------------"
    
    # Run the script as a module from the root directory
    python -m "$module"
    
    # Check if the script failed
    if [ $? -ne 0 ]; then
        echo "Error: $module failed. Stopping sequence."
        exit 1
    fi
    
    echo "Finished $module"
    echo "Waiting 5 seconds before next script..."
    sleep 5
done

echo "----------------------------------------------------------------"
echo "All listed binary benchmark scripts completed successfully!"
echo "----------------------------------------------------------------"
