import math
from typing import List, Optional

from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.ocr_error.loader import OCRErrorModelLoader
from surya.ocr_error.model.config import ID2LABEL
from surya.ocr_error.schema import OCRErrorDetectionResult
from surya.settings import settings
from surya.common.xla import mark_step


class OCRErrorPredictor(BasePredictor):
    model_loader_cls = OCRErrorModelLoader
    batch_size = settings.OCR_ERROR_BATCH_SIZE
    default_batch_sizes = {"cpu": 8, "mps": 8, "cuda": 64, "xla": 32}

    def __call__(self, texts: List[str], batch_size: Optional[int] = None):
        return self.batch_ocr_error_detection(texts, batch_size)

    def batch_ocr_error_detection(
        self, texts: List[str], batch_size: Optional[int] = None
    ):
        if batch_size is None:
            batch_size = self.get_batch_size()

        num_batches = math.ceil(len(texts) / batch_size)
        texts_processed = self.processor(
            texts, padding="longest", truncation=True, return_tensors="pt"
        )
        predictions = []
        for batch_idx in tqdm(
            range(num_batches),
            desc="Running OCR Error Detection",
            disable=self.disable_tqdm,
        ):
            start_idx, end_idx = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_input_ids = texts_processed.input_ids[start_idx:end_idx].to(
                self.model.device
            )
            batch_attention_mask = texts_processed.attention_mask[start_idx:end_idx].to(
                self.model.device
            )

            # Pad to batch size
            current_batch_size = batch_input_ids.shape[0]
            if settings.OCR_ERROR_STATIC_CACHE:
                batch_input_ids = self.pad_to_batch_size(batch_input_ids, batch_size)
                batch_attention_mask = self.pad_to_batch_size(
                    batch_attention_mask, batch_size
                )

            with settings.INFERENCE_MODE():
                pred = self.model(batch_input_ids, attention_mask=batch_attention_mask)

                logits = pred.logits.argmax(dim=1).cpu().tolist()[:current_batch_size]
                predictions.extend(logits)
            mark_step()

        return OCRErrorDetectionResult(
            texts=texts, labels=[ID2LABEL[p] for p in predictions]
        )
