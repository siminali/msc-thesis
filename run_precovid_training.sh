#!/bin/bash
# Pre-COVID Training Runner - Shell Script
# Usage examples for the pre-COVID training system

echo "========================================"
echo "Pre-COVID Training Runner"
echo "========================================"

# Set up environment variables
export PYTHONPATH="/Users/siminali/Desktop/Thesis Coding/src:/Users/siminali/Desktop/Thesis Coding"

# Change to the project directory
cd "/Users/siminali/Desktop/Thesis Coding"

# Function to check if file exists, use V2 if original exists
run_training() {
    if [ -f "train_precovid_models.py" ] && [ -f "train_precovid_models_v2.py" ]; then
        echo "Both original and V2 versions exist. Using V2 for safety..."
        python train_precovid_models_v2.py "$@"
    elif [ -f "train_precovid_models_v2.py" ]; then
        echo "Using V2 version..."
        python train_precovid_models_v2.py "$@"
    elif [ -f "train_precovid_models.py" ]; then
        echo "Using original version..."
        python train_precovid_models.py "$@"
    else
        echo "Error: No training script found!"
        exit 1
    fi
}

# Default training (all models)
echo "Starting default training (all models)..."
run_training --models all --epochs 50 --batch-size 32 --lr 1e-3

# Train only specific models (example)
echo ""
echo "Example: Training only zero and explicit models..."
# run_training --models zero explicit --epochs 100 --batch-size 64

# Train with different parameters (example)
echo ""
echo "Example: Training with custom parameters..."
# run_training --models llm --epochs 200 --pca-components 64 --hidden-dim 256

echo ""
echo "Training completed! Check checkpoints/precovid/ for results."
