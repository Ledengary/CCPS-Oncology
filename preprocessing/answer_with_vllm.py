#!/usr/bin/env python3
"""
Run zero-shot vLLM inference on a multiple-choice QA table (JSONL or CSV) and
emit:
  - <output-dir>/<model-name>/<stem>_answered.jsonl  (full per-row records + llm_output)

The input table must contain an `llm_input` column with the fully formatted
prompt (clinical note + question + options) and optionally a `system_prompt`
column. Temperature is 0.0 by default; the model is asked to return a single
letter.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-location", required=True, help="Input JSONL or CSV with llm_input column")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--llm-id", required=True, help="HF model id (e.g. Qwen/Qwen2.5-3B-Instruct)")
    p.add_argument("--llm-dir", default=None, help="Override local model directory")
    p.add_argument("--visible-cudas", required=True, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--gpu-memory", type=float, default=0.9)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--max-seq-len", type=int, default=1, help="Max output tokens (1 = single letter)")
    p.add_argument("--chat-template", default="openai")
    p.add_argument("--tokenizer-mode", default=None)
    p.add_argument("--quantization", default=None)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "utils"))
    from general import seed_everything                # noqa: E402
    from talk2llm import Talk2LLM                       # noqa: E402
    from utils.data_io import read_table, write_table   # noqa: E402

    seed_everything(args.seed)

    df = read_table(args.data_location)
    if args.limit:
        df = df.head(args.limit)
    if "llm_input" not in df.columns:
        raise ValueError("Input must contain an 'llm_input' column")

    has_sys = "system_prompt" in df.columns
    conversations = []
    for _, row in df.iterrows():
        msgs = []
        if has_sys:
            msgs.append({"role": "system", "content": str(row["system_prompt"])})
        msgs.append({"role": "user", "content": str(row["llm_input"])})
        conversations.append(msgs)

    llm = Talk2LLM(
        model_id=args.llm_dir or args.llm_id,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
        enforce_eager=False,
        tokenizer_mode=args.tokenizer_mode,
        quantization=args.quantization,
        the_seed=args.seed,
    )

    outs = llm.batch_chat_query(
        conversations,
        temperature=args.temp,
        max_tokens=args.max_seq_len,
        use_tqdm=True,
        chat_template_content_format=args.chat_template,
    )
    df["llm_output"] = outs

    model_name = args.llm_id.split("/")[-1]
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data_location).stem

    write_table(df, out_dir / f"{stem}_answered.jsonl")
    logging.info(f"Saved → {out_dir / f'{stem}_answered.jsonl'}")


if __name__ == "__main__":
    main()
