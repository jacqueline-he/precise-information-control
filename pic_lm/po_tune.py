#!/usr/bin/env python
# coding=utf-8

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import math
import os
import numpy as np
import random
import datasets
from datasets import load_dataset
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.prompt_utils import load_prompts, format_claims
from statistics import mean
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    get_scheduler,
)
from pic_lm.train_helpers import (
    DataCollatorForSeq2SeqDPO,
    concatenated_forward,
    dpo_loss,
    simpo_loss,
)
import wandb
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import deepspeed
from copy import deepcopy

logger = get_logger(__name__)


def prepare_deepspeed(accelerator, model):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if (
                hidden_size is not None
                and config_kwargs["zero_optimization"]["stage"] == 3
            ):
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size
                        * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10
                        * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def encode_sft_sample(example, tokenizer, max_seq_length, prompt_dict):
    instruction = example["source"]
    pic_type = example["pic_type"]
    formatted_claims = format_claims(example["claims"])
    template = "".join(prompt_dict[f"pic_{pic_type}_prompt"])
    instr_prompt = template.format(instruction=instruction, claims=formatted_claims)
    target_prompt = example["target"] + tokenizer.eos_token

    source_with_context = instr_prompt + target_prompt
    tokenized_source_with_context = tokenizer(
        source_with_context,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )

    context_input_ids = tokenized_source_with_context.input_ids
    context_labels = context_input_ids.clone()
    tokenized_prompt_with_context = tokenizer(
        instr_prompt, return_tensors="pt", max_length=max_seq_length, truncation=True
    )

    context_labels[:, : len(tokenized_prompt_with_context.input_ids[0])] = -100

    attention_mask = torch.ones_like(context_input_ids)
    return {
        "input_ids": context_input_ids.flatten(),
        "labels": context_labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_dpo_sample(example, tokenizer, max_seq_length, prompt_dict):
    pic_type = example["pic_type"]
    claims = example["claims"]
    instruction = example["instruction"]
    chosen_target = example["chosen"]
    rejected_target = example["rejected"]
    chosen_encoded = encode_sft_sample(
        {
            "source": instruction,
            "target": chosen_target,
            "claims": claims,
            "pic_type": pic_type,
        },
        tokenizer,
        max_seq_length,
        prompt_dict,
    )
    rejected_encoded = encode_sft_sample(
        {
            "source": instruction,
            "target": rejected_target,
            "claims": claims,
            "pic_type": pic_type,
        },
        tokenizer,
        max_seq_length,
        prompt_dict,
    )

    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_labels": chosen_encoded["labels"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_labels": rejected_encoded["labels"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
    }


def save_with_accelerate(accelerator, model, tokenizer, output_dir, use_lora=False):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=False
            )
    else:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )


@hydra.main(
    config_path="../conf/pic_lm",
    config_name="default_dpo",
    version_base="1.2",
)
def main(cfg: DictConfig):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if cfg.training.with_tracking:
        accelerator_log_kwargs["project_dir"] = cfg.run.output_dir
        accelerator_log_kwargs["log_with"] = ["wandb"]

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=cfg.training.timeout)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.run.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info(
        f"Global rank: {accelerator.process_index}, Local rank: {accelerator.local_process_index}, is_main: {accelerator.is_main_process}, is_local_main: {accelerator.is_local_main_process}"
    )

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        if cfg.run.output_dir is not None:
            os.makedirs(cfg.run.output_dir, exist_ok=True)
            OmegaConf.save(config=cfg, f=f"{cfg.run.output_dir}/config.yaml")

    # If passed along, set the training seed now.
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(cfg.model.sft_model_name_or_path)

    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name, use_fast=not cfg.model.use_slow_tokenizer
        )
    elif cfg.model.sft_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.sft_model_name_or_path, use_fast=not cfg.model.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    def load_model():
        if cfg.model.sft_model_name_or_path:
            if cfg.lora.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                device_index = accelerator.local_process_index
                device_map = {"": device_index}  # force data-parallel training.
                return AutoModelForCausalLM.from_pretrained(
                    cfg.model.sft_model_name_or_path,
                    from_tf=bool(".ckpt" in cfg.model.sft_model_name_or_path),
                    config=config,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True if cfg.model.use_flash_attn else False,
                )
            else:
                return AutoModelForCausalLM.from_pretrained(
                    cfg.model.sft_model_name_or_path,
                    from_tf=bool(".ckpt" in cfg.model.sft_model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=cfg.training.low_cpu_mem_usage,
                    use_flash_attention_2=True if cfg.model.use_flash_attn else False,
                    torch_dtype=torch.bfloat16,
                )
        else:
            logger.info("Training new model from scratch")
            return AutoModelForCausalLM.from_config(config)

    model = load_model()
    if cfg.loss.dpo_loss_type != "simpo":
        if not cfg.lora.use_lora:
            reference_model = load_model()
        else:
            reference_model = model
    else:
        reference_model = None

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "</s>",  # changed here from the original script so we do not introduce new token to vocabulary
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        assert tokenizer.pad_token is not None, "Tokenizer pad token is none"
    elif (
        isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        and tokenizer.pad_token is None
    ):
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert (
            num_added_tokens == 1
        ), "We detected no padding token but add_special_tokens did not add one."

    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
    if reference_model is not None:
        reference_embeddings = reference_model.get_input_embeddings()
        with deepspeed.zero.GatheredParameters(
            reference_embeddings.weight, modifier_rank=None
        ):
            if len(tokenizer) > reference_embeddings.weight.shape[0]:
                reference_model.resize_token_embeddings(len(tokenizer))

    if cfg.lora.use_lora:
        if cfg.lora.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    with accelerator.main_process_first():
        ds = load_dataset(cfg.dataset.name)
        if (
            cfg.dataset.max_num_samples is not None
            and len(ds["train"]) > cfg.dataset.max_num_samples
        ):
            logger.info(f"Truncating dataset to {cfg.dataset.max_num_samples} samples")
            ds["train"] = ds["train"].select(range(cfg.dataset.max_num_samples))
        dataset = ds["train"]

        encode_function = partial(
            encode_dpo_sample,
            tokenizer=tokenizer,
            max_seq_length=cfg.training.max_seq_length,
            prompt_dict=load_prompts(),
        )

        columns_to_keep = [
            "chosen_input_ids",
            "chosen_labels",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_labels",
            "rejected_attention_mask",
        ]
        processed_dataset = dataset.map(
            encode_function,
            batched=False,
            num_proc=cfg.training.preprocessing_num_workers,
            load_from_cache_file=not cfg.training.overwrite_cache,
            remove_columns=[
                col for col in dataset.column_names if col not in columns_to_keep
            ],
            desc="Tokenizing and reformatting input data",
        )

        processed_dataset.set_format(type="pt")
        processed_dataset = processed_dataset.filter(
            lambda example: (example["chosen_labels"] != -100).any()
        )
        processed_dataset = processed_dataset.filter(
            lambda example: (example["rejected_labels"] != -100).any()
        )

        train_dataset = processed_dataset

        if accelerator.is_main_process:
            lengths = [len(example["chosen_input_ids"]) for example in train_dataset]
            quartiles = np.percentile(lengths, [25, 50, 75])
            min_length = min(lengths)
            max_length = max(lengths)
            avg_length = mean(lengths)

            logger.info(f"Min length: {min_length}")
            logger.info(f"Max length: {max_length}")
            logger.info(f"Avg length: {avg_length:.2f}")
            logger.info(f"Q25: {quartiles[0]:.1f}")
            logger.info(f"Q50: {quartiles[1]:.1f}")
            logger.info(f"Q75: {quartiles[2]:.1f}")

    # Log a few random samples from the training set:
    logger.info("*" * 50)
    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        chosen_input_ids_tensor = train_dataset[index]["chosen_input_ids"]
        chosen_input_ids_detok = tokenizer.decode(
            chosen_input_ids_tensor, skip_special_tokens=True
        )
        logger.info(f"Detokenized sample chosen input: {chosen_input_ids_detok}")

        rejected_input_ids_tensor = train_dataset[index]["rejected_input_ids"]
        rejected_input_ids_detok = tokenizer.decode(
            rejected_input_ids_tensor, skip_special_tokens=True
        )
        logger.info(f"Detokenized sample rejected input: {rejected_input_ids_detok}")
    logger.info("*" * 50)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2SeqDPO(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        batch_size=cfg.run.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if cfg.lora.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.training.learning_rate,
            optim_bits=8 if cfg.lora.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=cfg.training.learning_rate
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.run.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = (
            cfg.training.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        cfg.training.max_train_steps
        if overrode_max_train_steps
        else cfg.training.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(
            num_training_steps_for_scheduler * cfg.training.warmup_ratio
        ),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if not cfg.lora.use_lora and reference_model is not None:
        reference_model = prepare_deepspeed(accelerator, reference_model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.run.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.training.max_train_steps = (
            cfg.training.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(
        cfg.training.max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = cfg.training.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if cfg.training.with_tracking and accelerator.is_local_main_process:
        experiment_config = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(experiment_config.get("lr_scheduler_type"), Enum):
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
        accelerator.init_trackers("pic_lm", experiment_config)

    # Train!
    total_batch_size = (
        cfg.run.per_device_train_batch_size
        * accelerator.num_processes
        * cfg.run.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.run.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.run.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(cfg.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.training.resume_from_checkpoint:
        if (
            cfg.training.resume_from_checkpoint is not None
            or cfg.training.resume_from_checkpoint != ""
        ):
            checkpoint_path = cfg.training.resume_from_checkpoint
            path = os.path.basename(cfg.training.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * cfg.run.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // cfg.run.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    episode = 0
    local_metrics = torch.zeros((20), device=accelerator.device)
    for epoch in range(starting_epoch, cfg.training.num_train_epochs):
        model.train()
        if (
            cfg.training.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        average_log_prob_loss_types = ["simpo", "dpo_norm"]
        average_log_prob = cfg.loss.dpo_loss_type in average_log_prob_loss_types

        for step, batch in enumerate(
            active_dataloader
        ):  # L958 in https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_tune.py
            episode += len(batch["chosen_input_ids"]) * accelerator.num_processes
            # dpo forward pass & loss
            with accelerator.accumulate(model):
                policy_chosen_logps, policy_rejected_logps, aux_loss = (
                    concatenated_forward(
                        model,
                        batch,
                        average_log_prob=average_log_prob,
                        output_router_logits=cfg.loss.load_balancing_loss,
                    )
                )  # `aux_loss` is only used when load_balancing_loss is used
                if (
                    cfg.loss.dpo_loss_type == "dpo"
                    or cfg.loss.dpo_loss_type == "dpo_norm"
                ):
                    with torch.no_grad():
                        if cfg.lora.use_lora:
                            with accelerator.unwrap_model(model).disable_adapter():
                                reference_chosen_logps, reference_rejected_logps, _ = (
                                    concatenated_forward(
                                        model, batch, average_log_prob=average_log_prob
                                    )
                                )
                        else:
                            reference_chosen_logps, reference_rejected_logps, _ = (
                                concatenated_forward(
                                    reference_model,
                                    batch,
                                    average_log_prob=average_log_prob,
                                )
                            )
                    losses, _, _ = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                        beta=cfg.loss.dpo_beta,
                        label_smoothing=cfg.loss.dpo_label_smoothing,
                    )
                elif cfg.loss.dpo_loss_type == "simpo":
                    losses, _, _ = simpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        beta=cfg.loss.dpo_beta,
                        gamma_beta_ratio=cfg.loss.dpo_gamma_beta_ratio,
                        label_smoothing=cfg.loss.dpo_label_smoothing,
                    )
                else:
                    raise ValueError(f"Invalid dpo loss type {cfg.loss.dpo_loss_type}.")

                loss = losses.mean()
                if cfg.loss.load_balancing_loss:
                    weighted_aux_loss = cfg.loss.load_balancing_weight * aux_loss
                    loss += weighted_aux_loss

                accelerator.backward(loss)  # L1010
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and cfg.training.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                ## Keep track of the loss at each logged step
                with torch.no_grad():
                    local_metrics[0] += loss
                    if (
                        cfg.loss.dpo_loss_type == "dpo"
                        or cfg.loss.dpo_loss_type == "dpo_norm"
                    ):
                        chosen_rewards = (
                            cfg.loss.dpo_beta
                            * (policy_chosen_logps - reference_chosen_logps)
                        ).mean()
                        rejected_rewards = (
                            cfg.loss.dpo_beta
                            * (policy_rejected_logps - reference_rejected_logps)
                        ).mean()
                        average_rewards = (chosen_rewards + rejected_rewards) / 2
                        accuracy = (chosen_rewards > rejected_rewards).float().mean()
                        margin = (chosen_rewards - rejected_rewards).mean()
                        local_metrics[1] += chosen_rewards
                        local_metrics[2] += rejected_rewards
                        local_metrics[3] += average_rewards
                        local_metrics[4] += accuracy
                        local_metrics[5] += margin
                    local_metrics[6] += policy_chosen_logps.mean()
                    local_metrics[7] += policy_rejected_logps.mean()
                    if cfg.loss.load_balancing_loss:
                        local_metrics[19] += weighted_aux_loss

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if (
                    cfg.training.logging_steps
                    and completed_steps % cfg.training.logging_steps == 0
                ):
                    # single all reduce to save time, avoiding per metric all reduce
                    global_metrics = accelerator.reduce(local_metrics, reduction="mean")
                    global_metrics /= (
                        cfg.run.gradient_accumulation_steps * cfg.training.logging_steps
                    )
                    global_metrics = global_metrics.tolist()
                    metrics_to_log = {
                        "training_step": completed_steps,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": episode / len(train_dataset),
                        "train_loss": global_metrics[0],
                        "logps/chosen": global_metrics[6],
                        "logps/rejected": global_metrics[7],
                    }
                    if (
                        cfg.loss.dpo_loss_type == "dpo"
                        or cfg.loss.dpo_loss_type == "dpo_norm"
                    ):
                        metrics_to_log.update(
                            {
                                "rewards/chosen": global_metrics[1],
                                "rewards/rejected": global_metrics[2],
                                "rewards/average": global_metrics[3],
                                "rewards/accuracy": global_metrics[4],
                                "rewards/margin": global_metrics[5],
                            }
                        )
                    if accelerator.is_main_process:
                        logger_str = f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {global_metrics[0]}"
                        if cfg.loss.load_balancing_loss:
                            logger_str += f" Aux Loss: {global_metrics[19]}"
                            metrics_to_log["aux_loss"] = global_metrics[19]
                        logger.info(logger_str)
                        if cfg.training.with_tracking:
                            accelerator.log(
                                metrics_to_log,
                                step=completed_steps,
                            )
                            wandb.log(metrics_to_log, step=completed_steps)
                    # Reset the local metrics
                    local_metrics.zero_()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if cfg.run.output_dir is not None:
                            output_dir = os.path.join(cfg.run.output_dir, output_dir)
                        save_with_accelerate(
                            accelerator, model, tokenizer, output_dir, cfg.lora.use_lora
                        )

                if completed_steps >= cfg.training.max_train_steps:
                    break

        if cfg.training.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if cfg.run.output_dir is not None:
                output_dir = os.path.join(cfg.run.output_dir, output_dir)
            save_with_accelerate(
                accelerator, model, tokenizer, output_dir, cfg.lora.use_lora
            )

    if cfg.run.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(cfg.run.output_dir)
        save_with_accelerate(
            accelerator, model, tokenizer, cfg.run.output_dir, cfg.lora.use_lora
        )

    if cfg.training.with_tracking:
        accelerator.end_training()
        if accelerator.is_local_main_process:
            wandb.finish()


if __name__ == "__main__":
    main()
