#!/bin/bash

# run_multiclass_benchmark.sh
# Script to run multi-class training benchmarks sequentially
# Run this from the project root directory

# Activate virtual environment if needed (uncomment and adjust path if necessary)
# source .venv/bin/activate

echo "Starting multi-class benchmark training sequence..."
echo "Note: PAN, UPerNet, PSPNet, SegFormer, Segmenter, ShallowCNN, FCN, DPT are skipped."
echo "Running remaining models: UNet, FPN, MAnet"

# List of scripts to run as modules (dot notation, no .py extension)
# Remaining models: UNet, FPN, MAnet (DPT excluded)
MODULES=(
    "benchmark_scripts.multi_case.train_Unet"
    "benchmark_scripts.multi_case.train_FPN"
    "benchmark_scripts.multi_case.train_MAnet"
)

# Create logs directory if it doesn't exist (handled by python scripts but good practice)
mkdir -p logs/benchmark/multi_case

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
echo "All listed benchmark scripts completed successfully!"
echo "----------------------------------------------------------------"
