#!/usr/bin/env python3
"""
Generate the three CORAL-derived oncology QA test tiers (Contextual, Synthesis,
Clinical Inference) using a four-stage GPT-5.1 pipeline: fact extraction →
blueprint generation → MCQ generation → verification. Tier selection swaps the
system prompts loaded from `prompts/<tier>.json`; the orchestration code is
shared.

The released test sets on PhysioNet were produced with this script and a fixed
set of GPT-5.1 calls. Running it against the same source CORAL records and the
same prompts will reproduce equivalent (not byte-identical) test sets, since
GPT-5.1 sampling is non-deterministic even at low temperature.

Inputs
------
--source-csv : path to the raw CORAL test CSV (`coral_test.csv`) with columns
               sidx, task, EHR, ground_truth_answer.
--tier       : one of {contextual, synthesis, clinical_inference}. Selects the
               prompt bundle.

Output
------
Per-record JSON files written under <output-dir>/<question-model>/<sidx>.json
containing the extracted facts, accepted blueprints, generated MCQs, and
verification verdicts. Downstream, these are aggregated into the four shipped
test CSVs (CORTEX_<tier>.csv).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coral_task_prompts import OncPrompt  # noqa: E402 — vendored from CORAL


PROMPT_DIR = Path(__file__).parent / "prompts"
VALID_TIERS = ["contextual", "synthesis", "clinical_inference"]


def load_prompts(tier: str) -> Dict[str, str]:
    if tier not in VALID_TIERS:
        raise ValueError(f"--tier must be one of {VALID_TIERS}, got {tier!r}")
    with (PROMPT_DIR / f"{tier}.json").open() as f:
        prompts = json.load(f)
    expected = {"fact_extraction", "blueprint_generation", "mcq_generation", "verification"}
    missing = expected - prompts.keys()
    if missing:
        raise ValueError(f"prompts/{tier}.json is missing keys: {sorted(missing)}")
    return prompts


# ----------------------------------------------------------------------------
# LLM helper
# ----------------------------------------------------------------------------

async def call_llm_json(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_tokens: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]
    try:
        response = await client.responses.create(
            model=model,
            reasoning={"effort": reasoning_effort},
            input=messages,
            max_output_tokens=max_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text and getattr(response, "output", None):
            for item in response.output:
                if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text" and hasattr(c, "text"):
                            output_text = c.text
                            break
                    break
        if not output_text:
            return {"status": "error", "error": "No output text from API"}
        try:
            return {"status": "success", "result": json.loads(output_text)}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"JSON parse failed: {e}", "raw_response": output_text}
    except Exception as e:
        return {"status": "error", "error": f"API call failed: {e}"}


# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------

async def extract_facts(client, row, model, max_tokens, effort, system_prompt):
    user_payload = {
        "task": row["task"],
        "original_prompt": OncPrompt().get_prompt(row["task"]),
        "ehr": row["EHR"],
        "ground_truth_answer": row["ground_truth_answer"],
    }
    resp = await call_llm_json(client, model, system_prompt, user_payload, max_tokens, effort)
    if resp.get("status") != "success":
        return []
    facts = resp.get("result", {}).get("facts", [])
    facts = [f for f in facts if f.get("difficulty", 1) >= 2][:20]
    return facts


async def generate_blueprints(client, row, facts, model, max_tokens, effort, max_per_record, system_prompt):
    if not facts:
        return []
    user_payload = {"task": row["task"], "ehr": row["EHR"], "facts": facts}
    resp = await call_llm_json(client, model, system_prompt, user_payload, max_tokens, effort)
    if resp.get("status") != "success":
        return []
    bps = resp.get("result", {}).get("question_blueprints", [])
    return [b for b in bps if b.get("difficulty") in ("medium", "hard")][:max_per_record]


async def generate_mcq(client, row, facts, blueprint, model, max_tokens, effort, system_prompt):
    user_payload = {"ehr": row["EHR"], "facts": facts, "question_blueprint": blueprint}
    resp = await call_llm_json(client, model, system_prompt, user_payload, max_tokens, effort)
    if resp.get("status") != "success":
        return {"status": "error", "error": resp.get("error", "Unknown error")}
    return resp.get("result", {})


async def verify_mcq(client, row, mcq, model, max_tokens, effort, system_prompt):
    user_payload = {"ehr": row["EHR"], "mcq": mcq}
    resp = await call_llm_json(client, model, system_prompt, user_payload, max_tokens, effort)
    if resp.get("status") != "success":
        return {"is_valid": False, "reason_if_invalid": resp.get("error", "Unknown error")}
    return resp.get("result", {})


def mcq_is_acceptable(mcq: Dict[str, Any], ver: Dict[str, Any]) -> bool:
    if not ver.get("is_valid", False):
        return False
    claimed = mcq.get("correct_answer")
    chosen = ver.get("model_selected_answer")
    if chosen != claimed:
        return False
    supported = [k for k, v in ver.get("option_evaluations", {}).items() if v == "supported"]
    return len(supported) == 1 and supported[0] == claimed


# ----------------------------------------------------------------------------
# Row orchestration
# ----------------------------------------------------------------------------

async def process_row(client, row, models, efforts, max_tokens, max_per_record, use_verifier, prompts, output_dir):
    sidx = row["sidx"]
    out_file = output_dir / f"{sidx}.json"
    if out_file.exists():
        return

    facts = await extract_facts(
        client, row, models["facts"], max_tokens, efforts["facts"], prompts["fact_extraction"]
    )
    if not facts:
        return

    bps = await generate_blueprints(
        client, row, facts, models["blueprints"], max_tokens, efforts["blueprints"],
        max_per_record, prompts["blueprint_generation"],
    )
    if not bps:
        return

    mcqs = []
    for bp in bps:
        mcq = await generate_mcq(
            client, row, facts, bp, models["question"], max_tokens, efforts["question"],
            prompts["mcq_generation"],
        )
        if mcq.get("status") == "error":
            continue
        if use_verifier:
            ver = await verify_mcq(
                client, row, mcq, models["verify"], max_tokens, efforts["verify"],
                prompts["verification"],
            )
            if not mcq_is_acceptable(mcq, ver):
                continue
            mcq["verification"] = ver
        mcqs.append({"blueprint": bp, "mcq": mcq})

    if mcqs:
        out_file.write_text(json.dumps(
            {"sidx": sidx, "task": row["task"], "mcqs": mcqs, "facts": facts}, indent=2
        ))


async def run(df, models, efforts, max_tokens, num_parallel, output_dir, api_key, max_per_record,
              use_verifier, prompts):
    client = AsyncOpenAI(api_key=api_key)
    model_output_dir = output_dir / models["question"]
    model_output_dir.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict(orient="records")
    sem = asyncio.Semaphore(num_parallel)

    async def wrapped(row_dict):
        async with sem:
            try:
                await process_row(
                    client, pd.Series(row_dict), models, efforts, max_tokens,
                    max_per_record, use_verifier, prompts, model_output_dir,
                )
            except Exception as e:
                tqdm.write(f"Error on sidx={row_dict.get('sidx')}: {e}")

    tasks = [wrapped(r) for r in rows]
    pbar = tqdm(total=len(tasks), desc="Generating MCQs", unit="row")
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()
    await client.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-csv", required=True, help="Path to coral_test.csv")
    p.add_argument("--tier", required=True, choices=VALID_TIERS,
                   help="Which test tier to generate")
    p.add_argument("--output-dir", required=True, help="Output JSON directory")
    p.add_argument("--model-facts", default="gpt-5-mini")
    p.add_argument("--model-blueprints", default="gpt-5.1")
    p.add_argument("--model-question", default="gpt-5.1")
    p.add_argument("--model-verify", default="gpt-5.1")
    p.add_argument("--reasoning-effort-facts", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--reasoning-effort-blueprints", default="high", choices=["low", "medium", "high"])
    p.add_argument("--reasoning-effort-question", default="high", choices=["low", "medium", "high"])
    p.add_argument("--reasoning-effort-verify", default="high", choices=["low", "medium", "high"])
    p.add_argument("--max-questions-per-record", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=25000)
    p.add_argument("--num-parallel", type=int, default=50)
    p.add_argument("--use-verifier", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many parent CORAL records (debug)")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key; falls back to OPENAI_API_KEY env var")
    args = p.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OpenAI API key not set (use --api-key or OPENAI_API_KEY)")

    df = pd.read_csv(args.source_csv)
    required = {"sidx", "task", "EHR", "ground_truth_answer"}
    if not required.issubset(df.columns):
        sys.exit(f"--source-csv is missing columns: {sorted(required - set(df.columns))}")
    if args.limit:
        df = df.head(args.limit)

    prompts = load_prompts(args.tier)
    models = {"facts": args.model_facts, "blueprints": args.model_blueprints,
              "question": args.model_question, "verify": args.model_verify}
    efforts = {"facts": args.reasoning_effort_facts, "blueprints": args.reasoning_effort_blueprints,
               "question": args.reasoning_effort_question, "verify": args.reasoning_effort_verify}

    print(f"Tier        : {args.tier}")
    print(f"Records     : {len(df)}")
    print(f"Models      : {models}")
    print(f"Effort      : {efforts}")
    print(f"Output dir  : {args.output_dir}")

    asyncio.run(run(
        df=df, models=models, efforts=efforts, max_tokens=args.max_tokens,
        num_parallel=args.num_parallel, output_dir=Path(args.output_dir),
        api_key=api_key, max_per_record=args.max_questions_per_record,
        use_verifier=args.use_verifier, prompts=prompts,
    ))


if __name__ == "__main__":
    main()
