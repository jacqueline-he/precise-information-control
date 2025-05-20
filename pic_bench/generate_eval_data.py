import os
import re
from functools import lru_cache
from vllm import LLM, SamplingParams
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.prompt_utils import format_claims, load_prompts
from utils.print_utils import fancy_print
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pic_bench.api import (
    get_anthropic_api_completion,
    get_openai_api_completion,
    fetch_api_completions_in_batches,
)
from datasets import load_dataset
import torch
from omegaconf import DictConfig
import hydra
from rich.console import Console
from rich.progress import Progress
import logging
import asyncio
from rich.logging import RichHandler

VALID_ENGINE_NAMES = ["vllm", "openai", "anthropic"]

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")
console = Console()


def split_thinking_output(text):
    parts = text.split("</think>")
    if len(parts) > 1:
        thinking = parts[0].strip()
        answer = parts[1].strip()
    else:
        thinking = None
        answer = text.strip()
    return thinking, answer


def create_prompts(data, PROMPT_DICT, is_thinking=False):
    prompts = []
    all_claims = []
    for d in data:
        instruction = d["instruction"]
        claims = d["claims"]
        formatted_claims = format_claims(claims)
        pic_type = d["pic_type"]
        template = "".join(PROMPT_DICT[f"pic_{pic_type}_prompt"])
        prompt = template.format(instruction=instruction, claims=formatted_claims)
        if is_thinking:
            # strip "[INST] and [/INST]"
            prompt = prompt.replace("[/INST]", "/think[/INST]<think>")
            # Add "\think" to the end of the prompt
        prompts.append(prompt)
        all_claims.append(claims)
    return prompts, all_claims


@lru_cache(maxsize=1)
def get_cached_vllm(
    model_name_or_path,
    download_dir,
    tensor_parallel_size,
    max_model_len,
    gpu_memory_utilization,
    max_num_batched_tokens,
    swap_space,
):
    llm = LLM(
        model=model_name_or_path,
        download_dir=download_dir,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        swap_space=swap_space,
        dtype=torch.bfloat16,
        disable_custom_all_reduce=True,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def get_task_data(task, max_instances=None):
    ds = load_dataset("jacquelinehe/pic-bench", split=task)
    if max_instances is not None:
        ds = ds.select(range(min(max_instances, len(ds))))
    return ds


def load_vllm(cfg):
    with console.status("[plum4]Loading LLM...", spinner="aesthetic"):
        llm, tokenizer = get_cached_vllm(
            model_name_or_path=cfg.model_name_or_path,
            download_dir=cfg.download_dir,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            swap_space=cfg.swap_space,
        )

    if cfg.is_thinking:
        sampling_params = SamplingParams(  # Add thinking sampling params
            max_tokens=cfg.thinking_sampling.max_tokens,
            temperature=cfg.thinking_sampling.temperature,
            repetition_penalty=cfg.thinking_sampling.repetition_penalty,
            seed=cfg.thinking_sampling.seed,
            top_p=cfg.thinking_sampling.top_p,
            top_k=cfg.thinking_sampling.top_k,
            min_p=cfg.thinking_sampling.min_p,
            presence_penalty=cfg.thinking_sampling.presence_penalty,
            stop=list(cfg.thinking_sampling.stop_tokens + [tokenizer.eos_token]),
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature,
            repetition_penalty=cfg.sampling.repetition_penalty,
            seed=cfg.sampling.seed,
            stop=list(cfg.sampling.stop_tokens + [tokenizer.eos_token]),
        )
    return llm, sampling_params


def load_api_client(engine_name, cfg):
    with console.status(
        f"[plum4]Loading {engine_name} API client...", spinner="aesthetic"
    ):
        if engine_name == "openai":
            client = AsyncOpenAI(api_key=cfg.api_key)
        elif engine_name == "anthropic":
            client = AsyncAnthropic(api_key=cfg.api_key)
        else:
            raise ValueError(
                f"Invalid engine name: {engine_name}. Must be one of {VALID_ENGINE_NAMES}"
            )
    return client


@hydra.main(
    config_path="../conf/eval", config_name="default_gen_vllm", version_base="1.2"
)
def generate_evals(cfg: DictConfig):
    engine_name = cfg.generation_engine.lower()
    if engine_name not in VALID_ENGINE_NAMES:
        raise ValueError(
            f"Invalid engine name: {engine_name}. Must be one of {VALID_ENGINE_NAMES}"
        )

    if engine_name == "vllm":
        llm, sampling_params = load_vllm(cfg)
    else:  # api LLM
        client = load_api_client(engine_name, cfg)

    all_tasks = cfg.tasks
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    all_task_str = "  ".join(all_tasks)

    console.print("[honeydew2]" + "*" * 100 + "[/honeydew2]")
    console.print(
        f"[deep_sky_blue2]Processing {len(all_tasks)} tasks:[/deep_sky_blue2] {all_task_str}"
    )
    console.print("[honeydew2]" + "*" * 100 + "[/honeydew2]")

    PROMPT_DICT = load_prompts()
    with Progress(console=console) as progress:
        task_id = progress.add_task("Processing...\n", total=len(all_tasks))

        for task in all_tasks:
            save_fp = os.path.join(save_dir, f"{task}.jsonl")
            if os.path.exists(save_fp) and not cfg.override:
                console.print(
                    f"[green]File {save_fp} already exists. Skipping...[/green]"
                )
                continue

            progress.update(task_id, description=f"[cyan]Processing: {task}\n")
            data = get_task_data(task, cfg.max_instances)
            prompts, all_claims = create_prompts(data, PROMPT_DICT, cfg.is_thinking)

            fancy_print(console, f"Ex. Prompt: {prompts[0]}")

            if engine_name == "vllm":
                outputs = llm.generate(prompts, sampling_params)
                generations = [o.outputs[0].text for o in outputs]
                tok_counts = [len(o.outputs[0].token_ids) for o in outputs]
            else:
                get_api_completion_fn = (
                    get_openai_api_completion
                    if engine_name == "openai"
                    else get_anthropic_api_completion
                )
                outputs = asyncio.run(
                    fetch_api_completions_in_batches(
                        client=client,
                        prompts=prompts,
                        model_name=cfg.model_name_or_path,
                        get_completion_fn=get_api_completion_fn,
                        batch_size=cfg.batch_size,
                        max_tokens=cfg.sampling.max_tokens,
                        temperature=cfg.sampling.temperature,
                        repetition_penalty=cfg.sampling.repetition_penalty,
                        seed=cfg.sampling.seed,
                        stop_tokens=(
                            list(cfg.sampling.stop_tokens)
                            if cfg.sampling.stop_tokens is not None
                            else None
                        ),
                    )
                )

                generations = [o["completion"] for o in outputs]
                tok_counts = [o["completion_tokens"] for o in outputs]
                if any(tok_counts):
                    total_tokens = sum(t for t in tok_counts if t is not None)
                    console.print(
                        f"[magenta]Total completion tokens for {engine_name}: {total_tokens}[/magenta]"
                    )

            fancy_print(console, f"Ex. Generation: {generations[0]}")

            if cfg.is_thinking:
                generated_data = []
                for prompt, claim, generation, tok_count in zip(
                    prompts, all_claims, generations, tok_counts
                ):
                    try:
                        think, final_ans = split_thinking_output(generation)
                    except Exception as e:
                        fancy_print(console, f"Error splitting thinking output: {e}")
                        think = None
                        final_ans = (
                            generation  # fallback to full output if parsing fails
                        )

                    generated_data.append(
                        {
                            "instruction": prompt,
                            "claims": claim,
                            "whole_output": generation,
                            "output": final_ans,
                            "think": think,
                            "completion_tok_count": tok_count,
                            "generator": cfg.model_name_or_path,
                            "is_thinking": True,
                        }
                    )
            else:
                generated_data = [
                    {
                        "instruction": prompt,
                        "claims": claim,
                        "output": generation,
                        "completion_tok_count": tok_count,
                        "generator": cfg.model_name_or_path,
                        "is_thinking": False,
                    }
                    for prompt, claim, generation, tok_count in zip(
                        prompts, all_claims, generations, tok_counts
                    )
                ]

            save_file_jsonl(generated_data, save_fp)
            fancy_print(console, f"Saved to {save_fp}")
            progress.advance(task_id)


if __name__ == "__main__":
    generate_evals()
