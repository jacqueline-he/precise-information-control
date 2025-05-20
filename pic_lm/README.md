## PIC-LM Training Framework

Our PIC-LM training framework consists of three steps:

1. Supervised fine-tuning (SFT) on diverse PIC-formatted instruction data;
2. Preference data construction using the SFT checkpoint and SFT dataset;
3. Direct preference optimization (DPO) on the constructed data.


### Scripts 
We release scripts for each part of the process. Please run all scripts from the root directory (i.e., in `precise-information-control/`). Our training scripts are heavily based on AI2's [open-instruct](https://github.com/allenai/open-instruct) repository. 

**SFT**
The SFT script in `examples/pic_lm_scripts/sft.sh` assumes 2 GPUs, with 1 batch size per GPU (and a total batch size of 32). The trained SFT checkpoint is saved locally to `out/pic_lm/pic-lm-8b-sft`. 

```bash
bash examples/pic_lm_scripts/sft.sh 
```

**Preference Data Creation**

The preference data construction script in `examples/pic_lm_scripts/create_pref_data.sh` processes the HF SFT dataset `jacquelinehe/pic-lm-sft-mixture`, using:
- **The reference LM** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **The SFT checkpoint** [jacquelinehe/Llama-3.1-PIC-LM-8B-SFT](https://huggingface.co/jacquelinehe/Llama-3.1-PIC-LM-8B-SFT) 

Our script supports individual modes for each step: `generate_perturbed_data`, `compute_normalized_log_prob`, `filter_data`, as well as `all` (runs all steps; this is the default mode). 

```bash
bash examples/pic_lm_scripts/create_pref_data.sh
```

**DPO**
The DPO script in `examples/pic_lm_scripts/dpo.sh` assumes 2 GPUs, with 1 batch size per GPU (and a total batch size of 128). This script initializes from the HF checkpoint `jacquelinehe/Llama-3.1-PIC-LM-8B-SFT` and saves the preference-tuned LM locally to `out/pic_lm/pic-lm-8b`. 

```bash
bash examples/pic_lm_scripts/dpo.sh 
```

### GPU memory issues?

Across both SFT and DPO, we perform _full_ fine-tuning on 8B models (i.e., all model parameters are updated), which may result in out-of-memory issues on GPUs with limited memory. If you encounter frequent CUDA OOM issues, we recommend trying the following in this order:

- Switch to DeepSpeed ZeRO-3 Offload by changing `--deepspeed_config_file` in the bash script to `${REPO_ROOT}/pic_lm/ds_configs/stage3_no_offloading_accelerate.conf`.
- Enable gradient checkpointing by setting `training.gradient_checkpointing=True`.
- Switch to low-rank adaptation (LoRA) training by setting `lora.use_lora=True`, and adjusting the LoRA hyperparameters.
- Switch to quantized LoRA training by setting `lora.use_qlora=True`, and adjusting the QLoRA hyperparameters.
- For preference optimization, we find that using the [SimPO](https://arxiv.org/abs/2405.14734) (Meng et al., 2024) loss leads to comparable performance while being much more compute- and memory-efficient, as the objective does not rely on a reference model. You can enable SimPO by setting `loss.dpo_loss_type=simpo`. 


