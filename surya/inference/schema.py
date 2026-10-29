from dataclasses import dataclass, field
from typing import Any, List, Optional

from PIL import Image


PROMPT_TYPE_LAYOUT = "layout"
PROMPT_TYPE_BLOCK = "block"
PROMPT_TYPE_TABLE_REC = "table_rec"
PROMPT_TYPE_HIGH_ACCURACY_BBOX = "high_accuracy_bbox"


@dataclass
class BatchInputItem:
    image: Image.Image
    prompt_type: str
    prompt: Optional[str] = None  # If set, overrides the default prompt for prompt_type
    max_tokens: Optional[int] = None
    temperature: Optional[float] = (
        None  # If set, overrides the backend default temperature
    )
    top_p: Optional[float] = None  # If set, overrides the backend default top_p
    request_logprobs: bool = False
    # vllm-native guided decoding — JSON schema, regex, or grammar string.
    # When set, the server constrains the decode tokens to match the schema.
    guided_json: Optional[dict] = None
    guided_regex: Optional[str] = None
    metadata: dict = field(default_factory=dict)  # Free-form, passes through to output


@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False
    # Mean of exp(logprob) across response tokens, if logprobs requested
    mean_token_prob: Optional[float] = None
    # Per-token logprobs (raw OpenAI-style content list), if requested - phase 2 use
    logprobs: Optional[List[Any]] = None
    # Server's finish_reason ("stop" | "length" | ...). "length" = the token cap cut the output.
    finish_reason: Optional[str] = None


@dataclass
class BatchOutputItem:
    raw: str
    token_count: int
    error: bool
    mean_token_prob: Optional[float] = None
    logprobs: Optional[List[Any]] = None
    metadata: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
