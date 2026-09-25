#!/bin/bash
# Universal training script for Wave Field LLM models
# Usage: ./train_model.sh <config_name> <num_gpus> [output_dir]

set -e

CONFIG_NAME=$1
NUM_GPUS=${2:-8}
OUTPUT_DIR=${3:-"models/${CONFIG_NAME}"}

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: $0 <config_name> [num_gpus] [output_dir]"
    echo "Example: $0 small 8 models/wavefield-small"
    exit 1
fi

CONFIG_PATH="model_zoo/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "========================================="
echo "Training Wave Field LLM Model"
echo "========================================="
echo "Config: $CONFIG_NAME"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    # Distributed training
    echo "Starting distributed training on $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        -m model_zoo.train_pipeline \
        --config "$CONFIG_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --num-gpus $NUM_GPUS
else
    # Single GPU training
    echo "Starting single GPU training..."
    python -m model_zoo.train_pipeline \
        --config "$CONFIG_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --num-gpus 1
fi

echo "========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================="

# Made with Bob
