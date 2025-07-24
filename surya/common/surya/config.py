from typing import Optional
from transformers import PretrainedConfig

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.encoder.config import SuryaEncoderConfig
from surya.common.surya.decoder.config import SuryaDecoderConfig


class SuryaModelConfig(S3DownloaderMixin, PretrainedConfig):
    model_type = "surya-multimodal-foundation"
    is_composition = True

    def __init__(
        self,
        vocab_size=65536,
        bbox_size=1025,
        blank_bbox_token_id=1025,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        image_token_id=3,
        register_token_ids=(4, 5, 6, 7),
        eoi_token_id=8,
        beacon_token_id=9,
        special_token_count=4,
        max_sequence_length=1536,
        special_ocr_tokens=None,
        vision_encoder=None,
        decoder=None,
        tasks: dict | None = None,
        bbox_embed_size: int = 64,
        num_register_tokens: int = 4,
        image_embed_encoding_size: int = 1024,
        image_embed_encoding_multiplier: int = 256,
        num_beacon_tokens: int = 1,
        beacon_token_interval: int = 4096,
        sliding_window: Optional[int] = None,
        multi_output_distance: int = 4,
        max_multi_out: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_encoder_decoder = False
        self.vocab_size = vocab_size
        self.bbox_size = bbox_size
        self.blank_bbox_token_id = blank_bbox_token_id
        self.image_token_id = image_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eoi_token_id = eoi_token_id
        self.beacon_token_id = beacon_token_id
        self.special_ocr_tokens = special_ocr_tokens
        self.special_token_count = special_token_count  # pad, bos, etc, tokens
        self.max_sequence_length = max_sequence_length
        self.tasks = tasks
        self.tie_word_embeddings = True
        self.bbox_embed_size = bbox_embed_size
        self.num_register_tokens = num_register_tokens
        self.register_token_ids = register_token_ids
        self.image_embed_encoding_size = image_embed_encoding_size
        self.image_embed_encoding_multiplier = image_embed_encoding_multiplier
        self.num_beacon_tokens = num_beacon_tokens
        self.beacon_token_interval = beacon_token_interval
        self.sliding_window = sliding_window
        self.multi_output_distance = multi_output_distance
        self.max_multi_out = max_multi_out

        if self.sliding_window is None:
            self.sliding_window = self.max_sequence_length

        if isinstance(vision_encoder, dict):
            vision_encoder = SuryaEncoderConfig(**vision_encoder)
        elif vision_encoder is None:
            vision_encoder = SuryaEncoderConfig()
        self.vision_encoder = vision_encoder

        if isinstance(decoder, dict):
            decoder = SuryaDecoderConfig(**decoder)
        elif decoder is None:
            decoder = SuryaDecoderConfig()
        self.decoder = decoder

        self.hidden_size = self.decoder.hidden_size

        self.patch_size = self.vision_encoder.spatial_patch_size
        self.merge_size = self.vision_encoder.spatial_merge_size
