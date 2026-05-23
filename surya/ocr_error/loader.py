from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.logging import get_logger
from surya.ocr_error.model.config import DistilBertConfig
from surya.ocr_error.model.encoder import DistilBertForSequenceClassification
from surya.ocr_error.tokenizer import DistilBertTokenizer
from surya.settings import settings

logger = get_logger()


class OCRErrorModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.OCR_ERROR_MODEL_CHECKPOINT

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE,
        attention_implementation: Optional[str] = None,
    ) -> DistilBertForSequenceClassification:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = DistilBertConfig.from_pretrained(self.checkpoint)
        model = (
            DistilBertForSequenceClassification.from_pretrained(
                self.checkpoint,
                dtype=dtype,
                config=config,
            )
            .to(device)
            .eval()
        )

        if settings.COMPILE_ALL or settings.COMPILE_OCR_ERROR:
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling detection model {self.checkpoint} from {DistilBertForSequenceClassification.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model = torch.compile(model, **compile_args)

        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> DistilBertTokenizer:
        return DistilBertTokenizer.from_pretrained(self.checkpoint)
