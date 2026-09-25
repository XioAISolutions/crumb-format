#!/bin/bash
# Evaluation script for Wave Field LLM models
# Usage: ./evaluate_model.sh <model_path> <output_file> [benchmarks...]

set -e

MODEL_PATH=$1
OUTPUT_FILE=$2
shift 2
BENCHMARKS=$@

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <model_path> <output_file> [benchmarks...]"
    echo "Example: $0 models/wavefield-small results/eval.json mmlu hellaswag"
    exit 1
fi

if [ -z "$BENCHMARKS" ]; then
    BENCHMARKS="mmlu hellaswag arc_easy arc_challenge winogrande"
fi

echo "========================================="
echo "Evaluating Wave Field LLM Model"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo "Benchmarks: $BENCHMARKS"
echo "========================================="

# Run evaluation
python -m model_zoo.evaluation.evaluator \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_FILE" \
    --benchmarks $BENCHMARKS \
    --device cuda \
    --batch-size 8

echo "========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================="

# Made with Bob
