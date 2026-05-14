#!/bin/bash
# Run zero-shot vLLM inference on all 4 QA datasets (train + 3 test tiers) for
# the 5 open-weight LLMs evaluated in the paper. Writes raw answers under
# data/answered/<model>/ and labeled JSONL (with binary correctness column)
# under data/predictions/<model>/.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/inference

declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
)

declare -A TIER_CSV=(
    [train]="data/train/ehrnoteqa_train_mcqa.jsonl"
    [contextual]="data/source/CORTEX_contextual.jsonl"
    [synthesis]="data/source/CORTEX_synthesis.jsonl"
    [clinical_inference]="data/source/CORTEX_clinical_inference.jsonl"
)

gpu_index=0
for model in "${MODELS[@]}"; do
    gpu="$gpu_index"
    safe_name=$(echo "$model" | tr '/' '-')
    for tier in train contextual synthesis clinical_inference; do
        csv="${TIER_CSV[$tier]}"
        [[ ! -f "$csv" ]] && { echo "skip $tier ($csv missing)"; continue; }

        out_answered="data/answered/${model##*/}"
        out_pred="data/predictions/${model##*/}"
        log="logs/inference/${safe_name}_${tier}.log"

        echo "[$model | $tier] GPU $gpu — log $log"

        python preprocessing/answer_with_vllm.py \
            --visible-cudas "$gpu" \
            --data-location "$csv" \
            --output-dir    "$out_answered" \
            --llm-id        "$model" \
            --dtype bfloat16 --temp 0.0 \
            --gpu-memory 0.9 --tensor-parallel 1 \
            --max-seq-len 1 \
            --chat-template openai \
            >> "$log" 2>&1

        # Convert answered JSONL → labeled JSONL
        ans_jsonl="${out_answered}/$(basename "$csv" .jsonl)_answered.jsonl"
        out_jsonl="${out_pred}/CORTEX_${tier}_labeled.jsonl"
        [[ "$tier" == "train" ]] && out_jsonl="${out_pred}/train_labeled.jsonl"
        python preprocessing/label_predictions.py \
            --in "$ans_jsonl" --out "$out_jsonl" \
            >> "$log" 2>&1
    done
    gpu_index=$(( (gpu_index + 1) % 5 ))
done
echo "Done — predictions under data/predictions/"
