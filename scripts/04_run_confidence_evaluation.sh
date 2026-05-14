#!/bin/bash
# Score every test record with each method and write per-record JSONs into
# data/confidence_scores/{model}/{method}/CORTEX_{tier}.json.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/evaluation

declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
)
declare -A TIER_SUFFIX=( [contextual]=v2 [synthesis]=mhr [clinical_inference]=l3 )
declare -A TIER_FILE=(
    [contextual]="CORTEX_contextual_labeled.jsonl"
    [synthesis]="CORTEX_synthesis_labeled.jsonl"
    [clinical_inference]="CORTEX_clinical_inference_labeled.jsonl"
)

gpu_index=0
for model in "${MODELS[@]}"; do
    gpu="$gpu_index"
    safe_name=$(echo "$model" | tr '/' '-')
    short=${model##*/}

    for tier in contextual synthesis clinical_inference; do
        # P(True) — runs the LLM live to elicit the self-evaluation logit
        python models/ptrue.py \
            --input-dir data/predictions \
            --model-name "$model" --model-id "$model" \
            --output-dir "data/confidence_scores/${short}/ptrue" \
            --llm-id "$model" --visible-cudas "$gpu" \
            --test-file "${TIER_FILE[$tier]}" --test-only \
            --dtype bfloat16 \
            > "logs/evaluation/${safe_name}_ptrue_${tier}.log" 2>&1

        # P(IK) — MLP probe over PIK representations
        python evaluation/pik_eval.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --representations-dir "representations/PIK_${TIER_SUFFIX[$tier]}" \
            --trained-models-dir   data/trained_models/pik \
            --results-dir          "data/confidence_scores/${short}/pik" \
            > "logs/evaluation/${safe_name}_pik_${tier}.log" 2>&1

        # SAPLMA — three layer variants
        for layer in final upper_middle middle; do
            sub=$([ "$layer" = "final" ] && echo "saplma_f" || ([ "$layer" = "upper_middle" ] && echo "saplma_um" || echo "saplma_m"))
            python evaluation/saplma_eval.py \
                --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
                --representations-dir "representations/SAPLMA_${TIER_SUFFIX[$tier]}" \
                --trained-models-dir data/trained_models/saplma \
                --results-dir        "data/confidence_scores/${short}/${sub}" \
                --layer-type "$layer" \
                > "logs/evaluation/${safe_name}_${sub}_${tier}.log" 2>&1
        done

        # CCPS — contrastive encoder + classifier head
        python evaluation/ccps_eval.py \
            --model-id "$model" --model-name "$model" --cuda-devices "$gpu" \
            --features-dir            "features/OrigPert_${TIER_SUFFIX[$tier]}" \
            --contrastive-model-path  data/trained_models/ccps/contrastive \
            --classifier-model-path   data/trained_models/ccps/classifier \
            --results-dir             "data/confidence_scores/${short}/ccps" \
            > "logs/evaluation/${safe_name}_ccps_${tier}.log" 2>&1
    done
    gpu_index=$(( (gpu_index + 1) % 5 ))
done
echo "Done — confidence scores under data/confidence_scores/"
