#!/usr/bin/env python
# coding=utf-8
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import logging
import math
import os
import datasets
from datasets import load_dataset
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from statistics import mean
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    get_scheduler,
    BitsAndBytesConfig,
)

import random
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from utils.prompt_utils import load_prompts, format_claims
from pic_lm.train_helpers import PICDataCollator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def encode_forward_sample(example, tokenizer, max_seq_length, prompt_dict):
    formatted_claims = format_claims(example["claims"])
    instruction = example["instruction"]
    response = example["response"]
    pic_type = example["pic_type"]

    template = "".join(prompt_dict[f"pic_{pic_type}_prompt"])
    instr_prompt = template.format(instruction=instruction, claims=formatted_claims)
    target_prompt = response + tokenizer.eos_token

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
    context_labels[:, : tokenized_prompt_with_context.input_ids.shape[1]] = -100

    return {
        "input_ids": context_input_ids.flatten(),
        "labels": context_labels.flatten(),
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
    config_name="default_sft",
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
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)

    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name, use_fast=not cfg.model.use_slow_tokenizer
        )
    elif cfg.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name_or_path, use_fast=not cfg.model.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if cfg.model.model_name_or_path:
        if cfg.lora.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name_or_path,
                from_tf=bool(".ckpt" in cfg.model.model_name_or_path),
                config=config,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if cfg.model.use_flash_attn else False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name_or_path,
                from_tf=bool(".ckpt" in cfg.model.model_name_or_path),
                config=config,
                low_cpu_mem_usage=cfg.training.low_cpu_mem_usage,
                use_flash_attention_2=True if cfg.model.use_flash_attn else False,
                torch_dtype=torch.bfloat16,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

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

        assert tokenizer.pad_token is not None, "Tokenizer pad token is none"
    elif (
        isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        and tokenizer.pad_token is None
    ):
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert (
            num_added_tokens == 1
        ), "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id  # 0419 new change

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
            encode_forward_sample,
            tokenizer=tokenizer,
            max_seq_length=cfg.training.max_seq_length,
            prompt_dict=load_prompts(),
        )

        columns_to_keep = ["input_ids", "labels"]
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
            lambda example: (example["labels"] != -100).any()
        )

        train_dataset = processed_dataset

        if accelerator.is_main_process:
            lengths = [len(example["input_ids"]) for example in train_dataset]
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
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"\nSample {index} of the training set: {train_dataset[index]}.")
        input_ids_tensor = train_dataset[index]["input_ids"]
        input_ids_detok = tokenizer.decode(input_ids_tensor, skip_special_tokens=False)
        logger.info(f"\nDetokenized sample input: {input_ids_detok}")
    logger.info("*" * 50)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=PICDataCollator(tokenizer),
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
    accelerator.wait_for_everyone()

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

    if cfg.training.with_tracking:
        experiment_config = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(experiment_config.get("lr_scheduler_type"), Enum):
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
        accelerator.init_trackers("pic_lm_sft", experiment_config)

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
        f"  Gradient accumulation steps = {cfg.run.gradient_accumulation_steps}"
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

        accelerator.print(f"Resuming from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
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

    for epoch in range(starting_epoch, cfg.training.num_train_epochs):
        model.train()
        total_loss = 0
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
        for _, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, cache_position=None, use_cache=False)
                # outputs = model(input_ids=batch['input_ids'],  labels=batch['labels'], cache_position=None, use_cache=False)
                loss = outputs.loss

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and cfg.training.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if (
                    cfg.training.logging_steps
                    and completed_steps % cfg.training.logging_steps == 0
                ):
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / cfg.run.gradient_accumulation_steps
                        / cfg.training.logging_steps
                    )

                    if accelerator.is_main_process:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss:.3f}"
                        )
                        if cfg.training.with_tracking:
                            metrics = {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                                "epoch": epoch,
                                "step": completed_steps,
                            }
                            accelerator.log(metrics, step=completed_steps)
                    total_loss = 0

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


if __name__ == "__main__":
    main()
