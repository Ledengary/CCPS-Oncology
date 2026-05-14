#!/bin/bash
# Train PIK, SAPLMA (3 layer variants), and CCPS (contrastive then classifier).
# Writes checkpoints to data/trained_models/{method}/<model_id>/.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/training

declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
)

gpu_index=0
for model in "${MODELS[@]}"; do
    gpu="$gpu_index"
    safe_name=$(echo "$model" | tr '/' '-')

    python models/pik/train.py \
        --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
        --representations-dir representations/PIK \
        --output-dir          data/trained_models/pik \
        > "logs/training/${safe_name}_pik.log" 2>&1

    for layer in final upper_middle middle; do
        python models/saplma/train.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --representations-dir representations/SAPLMA \
            --output-dir          data/trained_models/saplma \
            --layer-type "$layer" \
            > "logs/training/${safe_name}_saplma_${layer}.log" 2>&1
    done

    python models/ccps/contrastive_train.py \
        --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
        --features-dir features/OrigPert_v2 \
        --output-dir   data/trained_models/ccps/contrastive \
        > "logs/training/${safe_name}_ccps_contrastive.log" 2>&1

    python models/ccps/classifier_train.py \
        --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
        --features-dir         features/OrigPert_v2 \
        --contrastive-model-path data/trained_models/ccps/contrastive \
        --output-dir           data/trained_models/ccps/classifier \
        > "logs/training/${safe_name}_ccps_classifier.log" 2>&1

    gpu_index=$(( (gpu_index + 1) % 5 ))
done
echo "Done — trained classifiers under data/trained_models/"
