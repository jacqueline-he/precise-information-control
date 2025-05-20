import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import AsyncOpenAI, Timeout, APIError, APIConnectionError, RateLimitError
from anthropic import AsyncAnthropic, AnthropicError, APIStatusError
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn


# === Retry settings ===
openai_retry = retry(
    retry=retry_if_exception_type(
        (Timeout, APIError, APIConnectionError, RateLimitError)
    ),
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=300),
)

anthropic_retry = retry(
    retry=retry_if_exception_type(
        (AnthropicError, APIStatusError, APIConnectionError, RateLimitError)
    ),
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=1, max=300),
)


@openai_retry
async def get_openai_api_completion(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    max_tokens=4096,
    temperature=0.0,
    repetition_penalty=1.2,
    seed=42,
    stop_tokens=None,
    **kwargs
):
    completion = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=repetition_penalty,
        seed=seed,
        stop=stop_tokens,
    )
    return {
        "completion": completion.choices[0].message.content,
        "completion_tokens": completion.usage.completion_tokens,
    }


@anthropic_retry
async def get_anthropic_api_completion(
    client: AsyncAnthropic,
    prompt: str,
    model_name: str,
    max_tokens=2048,
    temperature=0.0,
    repetition_penalty=1.2,
    seed=42,
    stop_tokens=None,
    **kwargs
):
    completion = await client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop=stop_tokens,
    )
    return {
        "completion": completion.content[0].text,
        "completion_tokens": completion.usage.output_tokens,
    }


async def fetch_api_completions_in_batches(
    client, prompts, model_name, get_completion_fn, batch_size=50, **completion_args
):
    results = []
    total = len(prompts)

    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]

        tasks = [
            get_completion_fn(client, prompt, model_name, **completion_args)
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        await asyncio.sleep(1)  # To avoid rate limits or burst issues

    return results
