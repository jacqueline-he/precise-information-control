#!/usr/bin/env python
# coding=utf-8

import os
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.prompt_utils import format_claims, load_prompts
from utils.print_utils import fancy_print
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rich.progress import Progress
from rich.console import Console
import hydra
from omegaconf import DictConfig
import random
from log_prob_drop import compute_probability_drop_batched
from datasets import load_dataset

console = Console()


def get_min_max_indices(lst):
    min_index = min(enumerate(lst), key=lambda x: x[1])[0]
    max_index = max(enumerate(lst), key=lambda x: x[1])[0]
    return min_index, max_index


def clear_vllm(llm):
    del llm
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def drop_random_claims(cfg, claims, pic_type):
    full_drop_lower_fraction = cfg.full_drop_lower_fraction
    partial_drop_lower_fraction = cfg.partial_drop_lower_fraction
    drop_upper_fraction = cfg.drop_upper_fraction

    n = len(claims)
    if n <= 1:
        return claims

    if pic_type == "full":
        min_drop = max(1, int(n * full_drop_lower_fraction))
    else:
        min_drop = max(1, int(n * partial_drop_lower_fraction))

    max_drop = min(n - 1, int(n * drop_upper_fraction))
    if min_drop > max_drop:
        min_drop = max_drop

    num_to_drop = random.randint(min_drop, max_drop)
    indices_to_drop = set(
        random.sample(range(len(claims)), num_to_drop)
    )  # Pick random indices
    return [claim for i, claim in enumerate(claims) if i not in indices_to_drop]


def decode_prompts(generation_cfg, prompts):
    llm = LLM(
        model=generation_cfg.sft_model_path,
        download_dir=generation_cfg.download_dir,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=generation_cfg.max_tokens,
        top_k=generation_cfg.top_k,
        temperature=generation_cfg.temperature,
        repetition_penalty=generation_cfg.repetition_penalty,
        stop_tokens=generation_cfg.stop_tokens,
    )
    generations = llm.generate(prompts, sampling_params)
    clear_vllm(llm)
    return [gen.outputs[0].text for gen in generations]


def generate_perturbed_data(cfg):
    ds = load_dataset(cfg.hf_sft_dataset, cache_dir=cfg.download_dir)
    if cfg.generate_perturbed_data.max_num_samples is not None:
        ds = ds.select(range(cfg.generate_perturbed_data.max_num_samples))

    prompts, all_perturbed_claims, processed_data = [], [], []
    PROMPT_DICT = load_prompts()
    with Progress() as progress:
        total_samples = len(ds)
        task = progress.add_task("[cyan]Processing...", total=total_samples)
        for d in ds:
            instruction = d["instruction"]
            pic_type = d["pic_type"]
            claims = d["claims"]
            if len(claims) <= 1:
                continue  # Don't process
            perturbed_claims = drop_random_claims(
                cfg.generate_perturbed_data.random_drop, claims, pic_type
            )
            assert len(perturbed_claims) > 0

            formatted_claims = format_claims(claims, pic_type)
            prompt = PROMPT_DICT[pic_type]["prompt"].format(
                instruction=instruction, claims=formatted_claims
            )
            prompts.append(prompt)
            all_perturbed_claims.append(perturbed_claims)
            processed_data.append(d)
            progress.advance(task)

    generations = decode_prompts(cfg.generate_perturbed_data.generation, prompts)
    for pd, pc, gen in zip(processed_data, all_perturbed_claims, generations):
        pd["perturbed_response"] = gen
        pd["perturbed_claims"] = pc

    instr_data = []
    for d in processed_data:
        new_d = {}
        new_d["instruction"] = d["instruction"]
        new_d["original_claims"] = d["claims"]
        new_d["perturbed_claims"] = d["perturbed_claims"]
        new_d["original_response"] = d["response"]
        new_d["perturbed_response"] = d["perturbed_response"]
        new_d["pic_type"] = d["pic_type"]
        instr_data.append(new_d)
    save_file_jsonl(os.path.join(cfg.output_dir, "perturbed_data.jsonl"), instr_data)


def compute_normalized_log_prob(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.compute_normalized_log_probs.if_model_path,
        cache_dir=cfg.compute_normalized_log_probs.download_dir,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.compute_normalized_log_probs.if_model_path
    )
    model.eval()
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    data = load_jsonlines(os.path.join(cfg.output_dir, "perturbed_data.jsonl"))
    input_samples = []

    for d in data:
        instruction = d["instruction"]
        input_d = {
            "instruction": instruction,
            "original_response": d["original_response"],
            "perturbed_response": d["perturbed_response"],
        }
        input_samples.append(input_d)

    batch_size = cfg.compute_normalized_log_probs.batch_size
    all_results = []
    with Progress() as progress:

        total_batches = (len(input_samples) // batch_size) + 1
        task = progress.add_task("[cyan]Generating...", total=total_batches)

        for i in range(0, len(input_samples), batch_size):
            batch_samples = input_samples[i : i + batch_size]
            batch_instructions = [samp["instruction"] for samp in batch_samples]
            batch_original = [samp["original_response"] for samp in batch_samples]
            batch_perturbed = [samp["perturbed_response"] for samp in batch_samples]

            # Compute probability drop for the current batch.
            with torch.no_grad():
                batch_results = compute_probability_drop_batched(
                    model,
                    tokenizer,
                    batch_instructions,
                    batch_original,
                    batch_perturbed,
                    cfg.compute_normalized_log_probs.num_targeted_last_tokens,
                )

            all_results.extend(batch_results)
            progress.advance(task)

    for d, r in zip(data, all_results):
        d["prob_drop_results"] = r
    save_file_jsonl(data, os.path.join(cfg.output_dir, "log_prob_drop_scores.jsonl"))


def filter_data(cfg, data_dir):
    data_path = os.path.join(data_dir, "log_prob_drop_scores.jsonl")
    data = load_jsonlines(data_path)

    full_tau = cfg.filter_data.full_tau
    partial_tau = cfg.filter_data.partial_tau

    num_orig, num_perturbed = 0, 0
    for d in data:
        norm_score = d["prob_drop_results"]["norm_score"]
        original, perturbed = d["original_response"], d["perturbed_response"]
        pic_type = d["pic_type"]
        if (pic_type == "partial" and norm_score > partial_tau) or (
            pic_type == "full" and norm_score > full_tau
        ):
            d["chosen"], d["rejected"] = original, perturbed
            d["is_perturbed"] = False
            d["claims"] = d["original_claims"]
            d["tau_threshold"] = partial_tau if pic_type == "partial" else full_tau
            num_orig += 1
        else:
            d["chosen"], d["rejected"] = perturbed, original
            d["is_perturbed"] = True
            d["claims"] = d["perturbed_claims"]
            d["tau_threshold"] = partial_tau if pic_type == "partial" else full_tau
            num_perturbed += 1

    fancy_print(console, f"Number of original responses chosen: {num_orig}")
    fancy_print(console, f"Number of perturbed responses chosen: {num_perturbed}")
    new_fp = os.path.join(data_dir, "filtered_data.jsonl")
    save_file_jsonl(data, new_fp)


@hydra.main(
    config_path="../conf/pic_lm",
    config_name="default_pref_data",
    version_base="1.2",
)
def main(cfg: DictConfig):
    assert cfg.mode in [
        "all",
        "generate_perturbed_data",
        "compute_normalized_log_prob",
        "filter_data",
    ]

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.mode == "all" or cfg.mode == "generate_perturbed_data":
        generate_perturbed_data(cfg)
    if cfg.mode == "all" or cfg.mode == "compute_normalized_log_prob":
        compute_normalized_log_prob(cfg)
    if cfg.mode == "all" or cfg.mode == "filter_data":
        filter_data(cfg)


if __name__ == "__main__":
    main()
