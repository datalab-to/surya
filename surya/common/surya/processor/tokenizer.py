import html
import re
from typing import List, Union, Dict
import numpy as np
import torch
from tokenizers import AddedToken

from transformers import PreTrainedTokenizer, Qwen2Tokenizer as Qwen2OriginalTokenizer

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.schema import TASK_NAMES, TaskNames
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def create_token_regex(tokens):
    escaped_tokens = [re.escape(token) for token in tokens]
    escaped_tokens.sort(key=len, reverse=True)
    pattern = r"^(" + "|".join(escaped_tokens) + r")"
    regex = re.compile(pattern)
    return regex


class InnerOCRTokenizer:
    def __init__(
        self,
        special_tokens: Dict[str, list] | None = None,
        qwen_tokenizer: Qwen2OriginalTokenizer | None = None,
        **kwargs,
    ):
        self.qwen_tokenizer = qwen_tokenizer
        self.qwen_token_offset = len(qwen_tokenizer)

        all_special_tokens = special_tokens.get("all", [])
        self.SPECIAL_TOKEN_MAPPING = {}

        idx = 0
        for tag in all_special_tokens:
            if tag in self.SPECIAL_TOKEN_MAPPING:
                continue
            self.SPECIAL_TOKEN_MAPPING[tag] = (
                idx + self.qwen_token_offset
            )  # Assign token ID
            idx += 1

        self.REVERSE_SPECIAL_TOKEN_MAPPING = {
            v: k for k, v in self.SPECIAL_TOKEN_MAPPING.items()
        }
        self.SPECIAL_TOKEN_OFFSET = idx
        self.FORMAT_TAG_PATTERN = create_token_regex(special_tokens["formatting"])
        self.MATH_TAG_PATTERN = create_token_regex(special_tokens["math_external"])
        self.LAYOUT_TAG_PATTERN = create_token_regex(special_tokens["layout"])
        self.SYSTEM_TAG_PATTERN = create_token_regex(special_tokens.get("system", []))
        if not special_tokens.get("system", []):
            logger.warning("Warning: No system tokens found in special_tokens")

        self.MATH_TAG_START = "<math"
        self.MATH_END_TAG = "</math>"

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return (
            65536 + self.SPECIAL_TOKEN_OFFSET
        )  # The highest codepoint is 65535, but we add 1 to account for the 0-indexing

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        in_math = False
        text = html.unescape(text)  # Unescape html entities like &lt; in equations
        while text:
            # Look for EOS, PAD, etc. tokens
            match = self.SYSTEM_TAG_PATTERN.search(text)
            if match:
                tag = match.group(1)
                tokens.append(
                    self.SPECIAL_TOKEN_MAPPING[tag]
                )  # These are already offset
                text = text[match.end() :]
                continue

            # Look for layout tokens
            match = self.LAYOUT_TAG_PATTERN.search(text)
            if match:
                tag = match.group(1)
                tokens.append(
                    self.SPECIAL_TOKEN_MAPPING[tag]
                )   # Layout tokens are already offset
                text = text[match.end() :]
                continue

            # Check for math tags
            match = self.MATH_TAG_PATTERN.search(text)
            if match:
                # We found a tag
                tag = match.group(1)
                if tag.startswith(self.MATH_TAG_START):
                    in_math = True
                elif tag == self.MATH_END_TAG:
                    in_math = False
                tokens.append(
                    self.SPECIAL_TOKEN_MAPPING[tag]  # Special tokens are already offset
                )  # Use special token ID
                text = text[match.end() :]
                continue

            # Tokenize math content with qwen2 tokenizer
            if in_math:
                # If we're in a math block, check to see if we have a special math tag in the text
                math_end_position = text.find(self.MATH_END_TAG)
                math_str = text[:math_end_position]  # Gets the math content
                tokens += self.qwen_tokenizer(math_str)["input_ids"]
                text = text[math_end_position:]
                continue

            # Check for formatting tags
            match = self.FORMAT_TAG_PATTERN.search(text)
            if match:
                # We found a tag
                tag = match.group(1)
                tokens.append(
                    self.SPECIAL_TOKEN_MAPPING[tag]  # Special tokens are already offset
                )  # Use special token ID
                text = text[match.end() :]
                continue

            # General case, utf-16 tokenization
            utf_16_tokens = self.text_to_utf16_numbers(text[0])
            tokens += [
                t + self.SPECIAL_TOKEN_OFFSET + self.qwen_token_offset
                for t in utf_16_tokens
            ]
            text = text[1:]

        return tokens

    def text_to_utf16_numbers(self, text: str):
        """Converts text to UTF-16 encoded numbers."""
        utf16_bytes = text.encode(
            "utf-16le"
        )  # Little-endian to simplify byte order handling
        numbers = []

        for i in range(0, len(utf16_bytes), 2):
            # Combine two adjacent bytes into a single number
            number = utf16_bytes[i] + (utf16_bytes[i + 1] << 8)
            numbers.append(number)

        return numbers

    def utf16_numbers_to_text(self, numbers):
        """Converts UTF-16 numbers back to text."""
        byte_array = bytearray()
        for number in numbers:
            byte_array.append(number & 0xFF)  # Lower byte
            byte_array.append((number >> 8) & 0xFF)  # Upper byte

        try:
            text = byte_array.decode("utf-16le", errors="ignore")
        except Exception as e:
            logger.warning(f"Error decoding utf16: {e}")
            text = ""

        return text

    def __call__(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Dict[str, List[List[int]]]:
        """Tokenizes text and returns input IDs."""
        tokenized = []

        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = self._tokenize(text)
            tokenized.append(tokens)

        return {"input_ids": tokenized}

    def decode(self, token_ids, **kwargs):
        """Decodes token IDs back to text."""
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = token_ids.tolist()

        decoded_text = ""
        token_buffer = []
        decode_qwen = [False]

        def decode_buffer():
            nonlocal decoded_text, token_buffer, decode_qwen
            if token_buffer:
                if decode_qwen[0]:
                    decoded_text += self.qwen_tokenizer.decode(token_buffer)
                else:
                    token_buffer = [
                        t - self.SPECIAL_TOKEN_OFFSET - self.qwen_token_offset
                        for t in token_buffer
                    ]
                    decoded_text += self.utf16_numbers_to_text(token_buffer)

            token_buffer = []
            decode_qwen[0] = False

        for t in token_ids:
            if t < self.qwen_token_offset:
                # This is for math tags
                if token_buffer and token_buffer[-1] >= self.qwen_token_offset:
                    decode_buffer()
                token_buffer.append(t)
                decode_qwen[0] = True
            elif t >= self.SPECIAL_TOKEN_OFFSET + self.qwen_token_offset:
                if token_buffer and token_buffer[-1] < self.qwen_token_offset:
                    decode_buffer()
                token_buffer.append(t)  # We shift this down later on
                decode_qwen[0] = False
            elif t in self.REVERSE_SPECIAL_TOKEN_MAPPING:
                decode_buffer()
                decoded_text += self.REVERSE_SPECIAL_TOKEN_MAPPING[t]
                decode_qwen[0] = False
            else:
                raise ValueError(
                    f'Unexpected token value while decoding, got "{t}" in token_ids {token_ids}'
                )

        # Detokenize remaining tokens
        decode_buffer()

        return decoded_text


class Qwen2Tokenizer(S3DownloaderMixin, Qwen2OriginalTokenizer):
    pass


class SuryaOCRTokenizer(S3DownloaderMixin, PreTrainedTokenizer):
    def __init__(
        self,
        special_tokens: Dict[str, list] | None = None,
        model_checkpoint: str = settings.FOUNDATION_MODEL_CHECKPOINT,
        **kwargs,
    ):
        if special_tokens is None:
            special_tokens = dict()

        self.special_tokens = special_tokens

        self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(model_checkpoint)
        self.ocr_tokenizer = InnerOCRTokenizer(
            special_tokens=special_tokens, qwen_tokenizer=self.qwen_tokenizer
        )

        self.system_tokens = {
            v: self.ocr_tokenizer._tokenize(v)[0]
            for v in special_tokens.get("system", [])
        }
        self.SPECIAL_TOKEN_MAPPING = self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING

        super().__init__(**kwargs)

        self.qwen_offset = len(self.qwen_tokenizer)
        self.special_token_offset = (
            self.qwen_offset + self.ocr_tokenizer.SPECIAL_TOKEN_OFFSET
        )

    def get_vocab(self) -> Dict[str, int]:
        return self.qwen_tokenizer.get_vocab()

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        return self.qwen_tokenizer._add_tokens(
            new_tokens, special_tokens=special_tokens
        )

    @property
    def vocab_size(self):
        return self.ocr_tokenizer.vocab_size + self.qwen_offset

    def _tokenize(self, text: str, **kwargs):
        task = kwargs.get("task", TaskNames.ocr_with_boxes)
        assert task in TASK_NAMES, f"Invalid task: {task}"

        if task in [TaskNames.ocr_with_boxes, TaskNames.ocr_without_boxes, TaskNames.layout]:
            tokens = self.ocr_tokenizer._tokenize(text)
        else:
            tokens = self.qwen_tokenizer(text)["input_ids"]

        return tokens

    def __call__(
        self,
        texts: Union[str, List[str]],
        tasks: Union[str, List[str]] = None,
        **kwargs,
    ) -> Dict[str, List[List[int]]]:
        """Tokenizes text and returns input IDs."""
        tokenized = []

        if isinstance(texts, str):
            texts = [texts]
            assert isinstance(tasks, str), "Tasks must be a string if texts is a string"
            tasks = [tasks]

        if isinstance(texts, list):
            assert isinstance(tasks, list), "Tasks must be a list if texts is a list"

        for text, task in zip(texts, tasks):
            tokens = self._tokenize(text, task=task)
            tokenized.append(tokens)

        return {"input_ids": tokenized}

    def decode(self, token_ids, **kwargs):
        task_name = kwargs.get("task")
        assert task_name in TASK_NAMES, f"Invalid task: {task_name}"

        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = token_ids.tolist()

        if task_name in [TaskNames.ocr_with_boxes, TaskNames.ocr_without_boxes, TaskNames.layout]:
            decoded_text = self.ocr_tokenizer.decode(token_ids)
        else:
            decoded_text = self.qwen_tokenizer.decode(token_ids)

        return decoded_text
