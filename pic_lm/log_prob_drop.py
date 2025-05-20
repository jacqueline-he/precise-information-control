import torch
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence


def compute_avg_log_prob_batched(
    model, tokenizer, prompts, output_texts, num_targeted_last_tokens=20
):
    """
    Compute the average log probability for the last `num_targeted_last_tokens` tokens
    of each output_text given the corresponding prompt, processing examples in a batch.

    Args:
        model: The causal LM.
        tokenizer: Tokenizer associated with the model.
        prompts: List of prompt strings.
        output_texts: List of output strings.
        num_targeted_last_tokens: Number of tokens at the end of output_text to average.

    Returns:
        avg_log_probs: A list of average log probabilities, one per example.
    """
    # Tokenize prompts with padding.
    prompt_batch = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )
    prompt_batch = {k: v.to(model.device) for k, v in prompt_batch.items()}

    # For outputs we tokenize without adding padding (we need the individual lengths later).
    outputs_tokenized = [
        tokenizer.encode(text, return_tensors="pt").squeeze(0).to(model.device)
        for text in output_texts
    ]

    # Create concatenated inputs for each sample.
    # We leave outputs as-is; note that depending on your model,
    # you might need to remove the beginning-of-sequence token of outputs.
    concatenated_ids = []
    for i in range(len(prompts)):
        # Get the prompt tokens (for sample i) from the prompt batch.
        prompt_ids = prompt_batch["input_ids"][i]
        # Combine prompt and output tokens.
        concat = torch.cat([prompt_ids, outputs_tokenized[i]])
        concatenated_ids.append(concat)

    # Pad the concatenated sequences.
    concatenated_ids_padded = pad_sequence(
        concatenated_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # Compute attention mask from non-padding.
    attention_mask = (
        (concatenated_ids_padded != tokenizer.pad_token_id).long().to(model.device)
    )

    # Run the model on the full batch.
    with torch.no_grad():
        model_outputs = model(
            input_ids=concatenated_ids_padded, attention_mask=attention_mask
        )
    logits = model_outputs.logits  # shape: [batch_size, seq_len, vocab_size]

    avg_log_probs = []

    # Retrieve prompt lengths from the prompt batch (they may differ per example).
    prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()

    for i in range(len(prompts)):
        Lp = prompt_lens[i]
        # The model produced logits for the entire concatenated sequence.
        # Following the single-example logic, we consider the logits starting from position Lp-1.
        single_logits = logits[i, Lp - 1 :, :]  # shape: [output_len, vocab_size]

        # Retrieve the output tokens (we already have these in outputs_tokenized).
        output_ids = outputs_tokenized[i]

        # Determine how many tokens to target.
        nt = (
            num_targeted_last_tokens
            if num_targeted_last_tokens <= output_ids.shape[0]
            else output_ids.shape[0]
        )

        # The targeted logit slice is the last nt logits.
        targeted_logits = single_logits[-nt:, :]
        # Compute log probabilities.
        token_log_probs = F.log_softmax(targeted_logits, dim=-1)
        # Gather log probs for the actual tokens.
        # (Make sure output_ids are the target tokens.)
        targeted_ids = output_ids[-nt:]
        gathered = token_log_probs.gather(1, targeted_ids.unsqueeze(1)).squeeze(1)
        avg_log_prob = gathered.mean().item()
        avg_log_probs.append(avg_log_prob)

    return avg_log_probs


def compute_probability_drop_batched(
    model,
    tokenizer,
    instructions,
    original_responses,
    perturbed_responses,
    num_targeted_last_tokens=20,
):
    """
    Computes batched probability drop for a list of examples.

    For each example, computes:
       prob_drop = log p(y|x) - log p(y'|x)

    Args:
        model: The causal LM.
        tokenizer: Tokenizer.
        instructions: List of instruction strings.
        original_responses: List of original response strings.
        perturbed_responses: List of perturbed response strings.
        num_targeted_last_tokens: Number of tokens to consider for averaging.

    Returns:
        A list of dictionaries (one per sample), each with keys:
          "norm_score", "prob_drop", "log_prob_original", "log_prob_perturbed"
    """
    # Compute batched log probs for original responses.
    log_probs_original = compute_avg_log_prob_batched(
        model, tokenizer, instructions, original_responses, num_targeted_last_tokens
    )
    # Compute batched log probs for perturbed responses.
    log_probs_perturbed = compute_avg_log_prob_batched(
        model, tokenizer, instructions, perturbed_responses, num_targeted_last_tokens
    )

    results = []
    for lo, lp in zip(log_probs_original, log_probs_perturbed):
        prob_drop = lo - lp
        norm_score_val = normalized_score(prob_drop)
        results.append(
            {
                "norm_score": norm_score_val,
                "prob_drop": prob_drop,
                "log_prob_original": lo,
                "log_prob_perturbed": lp,
            }
        )
    return results


def compute_avg_log_prob(
    model, tokenizer, prompt, output_text, num_targeted_last_tokens=20
):
    """
    Compute the average log probability for the last `num_targeted_last_tokens` tokens
    of output_text given the prompt, using a causal LM.
    """
    # Encode the prompt and output text separately.
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = tokenizer.encode(output_text, return_tensors="pt")
    prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}
    output_ids = output_ids.to(model.device)

    # Concatenate prompt and output so the model conditions on the prompt.
    input_ids = torch.cat([prompt_inputs["input_ids"], output_ids], dim=1).to(
        model.device
    )

    # Run the model.
    with torch.no_grad():
        outputs = model(input_ids)

    # Get logits: shape [1, sequence_length, vocab_size].
    logits = outputs.logits
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # We only compute probabilities for the output tokens.
    # The logits corresponding to token t are produced at position t-1.
    # Extract logits for the output tokens.
    output_logits = logits[0, prompt_len - 1 : -1, :]

    # Get the target token ids for the output.
    target_ids = output_ids[0]
    # If we only want the last num_targeted_last_tokens:
    if num_targeted_last_tokens > len(target_ids):
        num_targeted_last_tokens = len(target_ids)
    targeted_ids = target_ids[-num_targeted_last_tokens:]

    # Similarly, get the corresponding logits.
    targeted_logits = output_logits[-num_targeted_last_tokens:, :]

    # Compute log probabilities for each token.
    log_probs = F.log_softmax(targeted_logits, dim=-1)
    # Gather the log probability for each token.
    token_log_probs = log_probs.gather(1, targeted_ids.unsqueeze(1)).squeeze(1)

    avg_log_prob = token_log_probs.mean().item()
    return avg_log_prob


def normalized_score(prob_drop):
    """
    Convert a log probability drop into a normalized score between 0 and 1.
    prob_drop: log p(y|x) - log p(y'|x)
    """
    ratio = math.exp(prob_drop)
    norm_score = ratio / (1 + ratio)
    return norm_score


def compute_probability_drop(
    model,
    tokenizer,
    instruction,
    original_response,
    perturbed_response,
    num_targeted_last_tokens=20,
):
    """
    Computes:
        prob-Drop = log p(y | x) - log p(y' | x)
    where x is the instruction, y is the original response, and y' is the perturbed response.
    """
    # Compute log probability for the original response.
    log_prob_original = compute_avg_log_prob(
        model, tokenizer, instruction, original_response, num_targeted_last_tokens
    )
    # Compute log probability for the perturbed response.
    log_prob_perturbed = compute_avg_log_prob(
        model, tokenizer, instruction, perturbed_response, num_targeted_last_tokens
    )

    # Probability drop:
    prob_drop = log_prob_original - log_prob_perturbed
    norm_score = normalized_score(prob_drop)
    return {
        "norm_score": norm_score,
        "prob_drop": prob_drop,
        "log_prob_original": log_prob_original,
        "log_prob_perturbed": log_prob_perturbed,
    }
