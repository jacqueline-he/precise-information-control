from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from prompts.pipeline_prompts import (
    DRAFT_BIRTHPLACE_PROMPT,
    VERIFY_BIRTHPLACE_PROMPT,
    INSTR_BIRTHPLACE_PROMPT,
    DRAFT_QAMPARI_PROMPT,
    VERIFY_QAMPARI_PROMPT,
)
from utils.prompt_utils import format_claims, load_prompts
from utils.text_utils import clean_for_ner
from utils.print_utils import fancy_print
import hydra
import logging
from omegaconf import DictConfig
from collections import defaultdict
from vllm import LLM, SamplingParams
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track
import gc
import os
import spacy
import torch
from vllm.distributed.parallel_state import destroy_model_parallel

spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")

VALID_MODES = ["generate_draft", "verify_draft_claims", "generate_final", "all"]
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("rich")
console = Console()

base_llm = None
final_lm = None


def get_base_llm(cfg):
    global base_llm
    if base_llm is not None:
        return base_llm
    with console.status("[plum4]Loading base LLM...", spinner="aesthetic"):
        base_llm = LLM(
            model=cfg.base_llm.model_name_or_path,
            download_dir=cfg.download_dir,
            tensor_parallel_size=cfg.base_llm.tensor_parallel_size,
            enforce_eager=cfg.base_llm.enforce_eager,
            max_model_len=cfg.base_llm.max_model_len,
            gpu_memory_utilization=cfg.base_llm.gpu_memory_utilization,
            dtype=torch.bfloat16,
        )
    return base_llm


def get_final_lm(cfg):
    global base_llm
    global final_lm

    if base_llm is not None:
        destroy_model_parallel()
        del base_llm
        gc.collect()
        torch.cuda.empty_cache()

    if final_lm is not None:
        return final_lm
    with console.status("[plum4]Loading final LM...", spinner="aesthetic"):
        final_lm = LLM(
            model=cfg.final_lm.model_name_or_path,
            download_dir=cfg.download_dir,
            tensor_parallel_size=cfg.final_lm.tensor_parallel_size,
            enforce_eager=cfg.final_lm.enforce_eager,
            max_model_len=cfg.final_lm.max_model_len,
            gpu_memory_utilization=cfg.final_lm.gpu_memory_utilization,
            dtype=torch.bfloat16,
        )
    return final_lm


def generate_draft(cfg, draft_fp):
    base_llm = get_base_llm(cfg)
    sampling_params = SamplingParams(
        temperature=cfg.sampling_params.generate_draft.temperature,
        max_tokens=cfg.sampling_params.generate_draft.max_tokens,
        repetition_penalty=cfg.sampling_params.generate_draft.repetition_penalty,
        stop=list(cfg.sampling_params.generate_draft.stop_tokens),
    )
    draft_prompt = (
        DRAFT_BIRTHPLACE_PROMPT if cfg.task == "birthplace" else DRAFT_QAMPARI_PROMPT
    )
    data = load_jsonlines(cfg.seed_data_path)
    prompts = []
    for d in track(data):
        if cfg.task == "birthplace":
            occ = d["occupation"]
            loc = d["loc"]
            prompts.append(draft_prompt.format(occupation=occ, loc=loc))
        else:
            prompts.append(draft_prompt.format(question=d["instruction"]))

    fancy_print(console, f"Example prompt: {prompts[0]}")
    outputs = base_llm.generate(prompts, sampling_params)
    fancy_print(console, f"Example output: {outputs[0].outputs[0].text.strip()}")
    final_resps = [output.outputs[0].text.strip() for output in outputs]
    draft_data = []
    for prompt, resp, d in track(
        zip(prompts, final_resps, data), description="Processing draft responses..."
    ):
        if cfg.task == "birthplace":
            draft_d = {
                "prompt": prompt,
                "draft_response": resp,
                "occupation": d["occupation"],
                "loc": d["loc"],
            }
            doc = nlp(resp)
            entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            entities = [entity.replace("'s", "").strip() for entity in entities]
            unique_entities = sorted(set(entities))

            if len(unique_entities) == 0:
                fancy_print(
                    console, f"Empty entities for {d['occupation']} in {d['loc']}"
                )

            draft_d["draft_entities"] = unique_entities
            draft_data.append(draft_d)
        else:
            draft_d = {
                "prompt": prompt,
                "draft_response": resp,
                "instruction": d["instruction"],
                "gold_answers": d["gold_answers"],
            }

            works_section = resp.split("\n\n")[0]
            # Split by comma, strip whitespace, and deduplicate while preserving order
            seen = set()
            deduped_works = []
            for item in works_section.split(","):
                work = item.strip()
                if work and work not in seen:
                    deduped_works.append(work)
                    seen.add(work)
            if not deduped_works:
                fancy_print(console, "error, empty set")
            draft_d["draft_entities"] = deduped_works
            draft_data.append(draft_d)

    os.makedirs(os.path.dirname(draft_fp), exist_ok=True)
    save_file_jsonl(draft_data, draft_fp)
    fancy_print(console, f"Saved draft output to {draft_fp}")


def verify_draft_claims(cfg, draft_fp, verify_fp):
    base_llm = get_base_llm(cfg)
    sampling_params = SamplingParams(
        temperature=cfg.sampling_params.verify_draft_claims.temperature,
        max_tokens=cfg.sampling_params.verify_draft_claims.max_tokens,
        repetition_penalty=cfg.sampling_params.verify_draft_claims.repetition_penalty,
        n=cfg.sampling_params.verify_draft_claims.n,
    )
    draft_data = load_jsonlines(draft_fp)
    prompts, ents_to_inds, all_entities = [], [], []
    for i, d in enumerate(draft_data):
        entities = d["draft_entities"]
        all_entities.extend(entities)
        inds = [int(i)] * len(entities)
        ents_to_inds.extend(list(inds))

        if cfg.task == "birthplace":
            occupation = d["occupation"]
            for entity in entities:
                prompts.append(
                    VERIFY_BIRTHPLACE_PROMPT.format(
                        entity=entity, occupation=occupation
                    )
                )
        else:
            question = d["instruction"]
            for entity in entities:
                prompts.append(
                    VERIFY_QAMPARI_PROMPT.format(question=question, answer=entity)
                )

    outputs = base_llm.generate(prompts, sampling_params)
    entity_resp_map = defaultdict(list)
    for entity, i, output in zip(all_entities, ents_to_inds, outputs):
        generations = [o.text.strip() for o in output.outputs]
        entity_resp_map[(i, entity)].extend(generations)

    # Initialize field in each data item
    for d in draft_data:
        d["verified_entities"] = []

    def check_equiv(resp, loc):
        resp = resp.lower()
        loc = loc.lower()
        return loc in resp

    for (i, entity), responses in entity_resp_map.items():
        if cfg.task == "birthplace":
            num_true = sum(check_equiv(r, draft_data[i]["loc"]) for r in responses)
        else:
            num_true = sum(r.strip().lower() == "true" for r in responses)
        is_true = num_true >= cfg.k
        draft_data[i]["verified_entities"].append((entity, is_true, responses))

    for d in track(draft_data):
        d["correct_entities"] = [
            entity for entity, is_true, _ in d["verified_entities"] if is_true
        ]

    save_file_jsonl(draft_data, verify_fp)
    fancy_print(console, f"Saved verifications to {verify_fp}")


def generate_final(cfg, verify_fp, final_fp):
    final_lm = get_final_lm(cfg)
    sampling_params = SamplingParams(
        temperature=cfg.sampling_params.generate_final.temperature,
        max_tokens=cfg.sampling_params.generate_final.max_tokens,
        repetition_penalty=cfg.sampling_params.generate_final.repetition_penalty,
        seed=cfg.sampling_params.generate_final.seed,
    )

    PROMPT_DICT = load_prompts()
    template = "".join(PROMPT_DICT[f"pic_{cfg.pic_type}_prompt"])
    data = load_jsonlines(verify_fp)
    prompts = []

    for d in track(data):
        entities = d["correct_entities"]
        if cfg.task == "birthplace":
            claims = format_claims(
                [
                    f"{entity} is a {d['occupation']} born in {d['loc']}."
                    for entity in entities
                ]
            )
            instruction = INSTR_BIRTHPLACE_PROMPT.format(
                occupation=d["occupation"], loc=d["loc"]
            )
        else:
            claims = format_claims(
                [
                    f"{entity} is a correct answer to this question."
                    for entity in entities
                ]
            )
            instruction = d["instruction"].split("?")[0] + "?"

        prompt = template.format(instruction=instruction, claims=claims)
        prompts.append(prompt)

    fancy_print(console, f"Ex. Prompt: {prompts[0]}")
    outputs = final_lm.generate(prompts, sampling_params)
    final_resps = [output.outputs[0].text.strip() for output in outputs]
    fancy_print(console, f"Ex. Generation: {final_resps[0]}")

    for d, resp in zip(data, final_resps):
        d["response"] = resp
        d["generator"] = cfg.final_lm.model_name_or_path

        if (
            cfg.task == "birthplace" and cfg.do_fast_eval
        ):  # do fast approximate factual precision eval
            resp = clean_for_ner(resp)
            doc = nlp(resp)
            pred_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            pred_entities = [
                entity.replace("'s", "").strip() for entity in pred_entities
            ]
            pred_entities = list(set(pred_entities))  # Deduplicate

            correct_entities = set(d["correct_entities"])
            matched_entities = [e for e in pred_entities if e in correct_entities]

            # Precision calculation
            precision = (
                len(matched_entities) / len(pred_entities) if pred_entities else 0.0
            )

            # Save to data
            d["fast_eval"] = {
                "predicted_entities": pred_entities,
                "matched_entities": matched_entities,
                "precision": precision,
            }
    os.makedirs(os.path.dirname(final_fp), exist_ok=True)
    save_file_jsonl(data, final_fp)
    fancy_print(console, f"Saved final output to {final_fp}")

    if cfg.task == "birthplace" and cfg.do_fast_eval:
        precs = [d["fast_eval"]["precision"] for d in data]
        avg_prec = sum(precs) / len(precs) if precs else 0.0
        perfect_precs = sum(1 for p in precs if p == 1.0)
        perfect_percentage = (perfect_precs / len(precs) * 100) if precs else 0.0
        console.print(
            f"[plum4]Average factual precision for birthplace: {avg_prec:.2f} | Perfect precision: {perfect_percentage:.1f}%[/plum4]"
        )


@hydra.main(
    config_path="../conf/use_cases",
    config_name="default_pipeline_birthplace",
    version_base="1.2",
)
def pipeline(cfg: DictConfig):
    mode = cfg.mode
    assert mode in VALID_MODES, f"{mode} should be one of {VALID_MODES}"
    assert cfg.task == "birthplace" or cfg.task == "qampari"

    draft_fp = cfg.draft_fp
    verify_fp = cfg.verify_fp
    final_fp = cfg.final_fp

    if mode == "all" or mode == "generate_draft":
        if not os.path.exists(draft_fp) or cfg.override_generate_draft:
            generate_draft(cfg, draft_fp)  # 1. Generate draft output
    if mode == "all" or mode == "verify_draft_claims":
        if not os.path.exists(verify_fp) or cfg.override_verify_draft_claims:
            verify_draft_claims(
                cfg, draft_fp, verify_fp
            )  # 2. Draft verification questions, 3. Execute verification questions
    if mode == "all" or mode == "generate_final":
        if not os.path.exists(final_fp) or cfg.override_generate_final:
            generate_final(cfg, verify_fp, final_fp)  # 4. Generate final response


if __name__ == "__main__":
    pipeline()
