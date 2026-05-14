#!/bin/bash
# Extract the per-method hidden states required for each confidence estimator:
#   • PIK     — final hidden state of the last input token
#   • SAPLMA  — final-token hidden state of the model's answer, at 3 layers
#                (final, upper_middle, middle)
#   • CCPS    — original + perturbed hidden states and logits (PEI radius 20, 5 steps)
#
# Outputs into representations/{method}/{model_id}/{split}/{layer}/*.npy(z).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/extraction representations

declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
)

# Tier suffix used by the legacy extractors:
declare -A TIER_SUFFIX=( [contextual]=v2 [synthesis]=mhr [clinical_inference]=l3 )
declare -A TIER_TEST_FILE=(
    [contextual]="CORTEX_contextual_labeled.jsonl"
    [synthesis]="CORTEX_synthesis_labeled.jsonl"
    [clinical_inference]="CORTEX_clinical_inference_labeled.jsonl"
)

gpu_index=0
for model in "${MODELS[@]}"; do
    gpu="$gpu_index"
    safe_name=$(echo "$model" | tr '/' '-')

    # -------- PIK and SAPLMA on the contextual training corpus --------
    python models/pik/extract_hidden_states.py \
        --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
        --input-dir  data/predictions \
        --representations-dir representations/PIK \
        --train-file train_labeled.jsonl \
        --test-file  CORTEX_contextual_labeled.jsonl \
        > "logs/extraction/${safe_name}_pik.log" 2>&1

    python models/saplma/extract_hidden_states.py \
        --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
        --input-dir  data/predictions \
        --representations-dir representations/SAPLMA \
        --train-file train_labeled.jsonl \
        --test-file  CORTEX_contextual_labeled.jsonl \
        > "logs/extraction/${safe_name}_saplma_contextual.log" 2>&1

    # SAPLMA on the harder test tiers (test-only)
    for tier in synthesis clinical_inference; do
        suffix="${TIER_SUFFIX[$tier]}"
        python models/saplma/extract_hidden_states.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --input-dir data/predictions \
            --representations-dir "representations/SAPLMA_${suffix}" \
            --test-file "${TIER_TEST_FILE[$tier]}" \
            --test-only \
            > "logs/extraction/${safe_name}_saplma_${tier}.log" 2>&1
    done

    # -------- CCPS: original + perturbed hidden states, all 3 tiers --------
    for tier in contextual synthesis clinical_inference; do
        suffix="${TIER_SUFFIX[$tier]}"
        python models/ccps/extract_hidden_states.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --input-dir data/predictions \
            --representations-dir "representations/OrigPert_${suffix}" \
            --test-file "${TIER_TEST_FILE[$tier]}" \
            --test-only --pei-radius 20.0 --pei-steps 5 --dtype float16 \
            > "logs/extraction/${safe_name}_ccps_${tier}.log" 2>&1

        python models/ccps/extract_features.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --representations-dir "representations/OrigPert_${suffix}" \
            --output-dir          "features/OrigPert_${suffix}" \
            --dtype float16 --test-only \
            > "logs/extraction/${safe_name}_ccps_${tier}_features.log" 2>&1
    done
    gpu_index=$(( (gpu_index + 1) % 5 ))
done
echo "Done — hidden states under representations/, CCPS features under features/"
