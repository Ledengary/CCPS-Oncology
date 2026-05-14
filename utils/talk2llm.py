"""
Thin wrapper around vLLM's LLM and SamplingParams that the inference scripts
use. Kept minimal — every option here maps directly to a vLLM constructor or
SamplingParams argument.
"""

import os

import torch
from vllm import LLM, SamplingParams


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Talk2LLM:
    def __init__(
        self,
        model_id,
        dtype=None,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        enforce_eager=None,
        task="auto",
        tokenizer_mode=None,
        config_format=None,
        quantization=None,
        load_format=None,
        the_seed=23,
    ):
        chosen_dtype = _DTYPE_MAP.get(dtype) if dtype else None
        print("Visible CUDAs for vLLM:", os.environ.get("CUDA_VISIBLE_DEVICES"))

        llm_kwargs = dict(
            model=model_id,
            task=task,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            seed=the_seed,
            trust_remote_code=True,
        )
        if chosen_dtype is not None:
            llm_kwargs["dtype"] = chosen_dtype
        for k, v in dict(
            tokenizer_mode=tokenizer_mode,
            load_format=load_format,
            config_format=config_format,
            quantization=quantization,
        ).items():
            if v is not None:
                llm_kwargs[k] = v

        self.llm = LLM(**llm_kwargs)

    def batch_chat_query(
        self,
        conversations,
        temperature=0.0,
        max_tokens=100,
        use_tqdm=True,
        chat_template_content_format="openai",
    ):
        if isinstance(temperature, list):
            assert len(temperature) == len(conversations)
            sp = [SamplingParams(temperature=t, max_tokens=max_tokens) for t in temperature]
        else:
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outs = self.llm.chat(
            messages=conversations,
            sampling_params=sp,
            use_tqdm=use_tqdm,
            chat_template_content_format=chat_template_content_format,
        )
        return [o.outputs[0].text.strip() for o in outs]
