"""Shared OpenAI-compatible chat completions client. Used by vllm + llama.cpp.

Both servers expose `/v1/chat/completions` with the same request/response shape,
so this module is the single point of HTTP contact for both backends.
"""

from __future__ import annotations

import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from PIL import Image

from surya.inference.prompts import PROMPT_MAPPING
from surya.inference.schema import (
    BatchInputItem,
    BatchOutputItem,
    GenerationResult,
)
from surya.inference.util import detect_repeat_token, scale_to_fit
from surya.logging import get_logger
from surya.settings import settings
from tqdm import tqdm

logger = get_logger()


def encode_image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_messages(image: Image.Image, prompt: str):
    image_b64 = encode_image_b64(image)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _mean_token_prob(logprobs_content) -> Optional[float]:
    if not logprobs_content:
        return None
    probs = []
    for tok in logprobs_content:
        lp = (
            tok.get("logprob")
            if isinstance(tok, dict)
            else getattr(tok, "logprob", None)
        )
        if lp is None:
            continue
        probs.append(math.exp(lp))
    if not probs:
        return None
    return sum(probs) / len(probs)


def _generate_one(
    item: BatchInputItem,
    client,
    model_name: str,
    max_tokens_default: int,
    temperature: float,
    top_p: float,
    timeout: float,
    request_logprobs_default: bool,
) -> GenerationResult:
    prompt = item.prompt or PROMPT_MAPPING[item.prompt_type]
    image = scale_to_fit(item.image)
    messages = _build_messages(image, prompt)

    max_tokens = item.max_tokens or max_tokens_default
    request_logprobs = item.request_logprobs or request_logprobs_default
    temp = item.temperature if item.temperature is not None else temperature
    tp = item.top_p if item.top_p is not None else top_p

    kwargs = dict(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
        top_p=tp,
        timeout=timeout,
    )
    if request_logprobs:
        kwargs["logprobs"] = True

    # Structured output: prefer OpenAI-standard response_format (works on both
    # vllm and llama.cpp). Fall back to vllm's extra_body for guided_regex.
    if item.guided_json is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": item.guided_json,
                "strict": True,
            },
        }
    if item.guided_regex is not None:
        kwargs.setdefault("extra_body", {})["guided_regex"] = item.guided_regex

    try:
        completion = client.chat.completions.create(**kwargs)
        raw = completion.choices[0].message.content or ""
        token_count = completion.usage.completion_tokens if completion.usage else 0
        mean_p = None
        logprobs_content = None
        if request_logprobs:
            choice = completion.choices[0]
            lp = getattr(choice, "logprobs", None)
            if lp is not None:
                content = getattr(lp, "content", None)
                if content is not None:
                    logprobs_content = [
                        c.model_dump() if hasattr(c, "model_dump") else c
                        for c in content
                    ]
                    mean_p = _mean_token_prob(content)
        return GenerationResult(
            raw=raw,
            token_count=token_count,
            error=False,
            mean_token_prob=mean_p,
            logprobs=logprobs_content,
        )
    except Exception as e:
        logger.warning(f"Inference error: {e}")
        return GenerationResult(raw="", token_count=0, error=True)


def _should_retry(
    result: GenerationResult,
    retries: int,
    max_retries: int,
) -> bool:
    if retries >= max_retries:
        return False
    if result.error:
        return True
    has_repeat = detect_repeat_token(result.raw) or (
        len(result.raw) > 50 and detect_repeat_token(result.raw, cut_from_end=50)
    )
    return has_repeat


def chat_completions_batch(
    batch: List[BatchInputItem],
    client,
    model_name: str,
    max_tokens_default: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.1,
    timeout: float = 600.0,
    max_workers: Optional[int] = None,
    max_retries: int = 3,
    request_logprobs_default: bool = True,
) -> List[BatchOutputItem]:
    """Run a batch of items through the chat completions endpoint with concurrent workers."""
    if not batch:
        return []
    if max_workers is None:
        max_workers = min(64, len(batch))

    def _process(item: BatchInputItem) -> BatchOutputItem:
        result = _generate_one(
            item,
            client=client,
            model_name=model_name,
            max_tokens_default=max_tokens_default,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            request_logprobs_default=request_logprobs_default,
        )
        retries = 0
        while _should_retry(result, retries, max_retries):
            backoff = 1.5 * (retries + 1) if result.error else 0
            if backoff:
                time.sleep(backoff)
            retry_temp = min(temperature + 0.2 * (retries + 1), 0.8)
            retry_top_p = 0.95 if not result.error else top_p
            result = _generate_one(
                item,
                client=client,
                model_name=model_name,
                max_tokens_default=max_tokens_default,
                temperature=retry_temp,
                top_p=retry_top_p,
                timeout=timeout,
                request_logprobs_default=request_logprobs_default,
            )
            retries += 1
        return BatchOutputItem(
            raw=result.raw,
            token_count=result.token_count,
            error=result.error,
            mean_token_prob=result.mean_token_prob,
            logprobs=result.logprobs,
            metadata=item.metadata,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(_process, batch),
                total=len(batch),
                desc="Recognizing Text",
                disable=settings.DISABLE_TQDM,
            )
        )
