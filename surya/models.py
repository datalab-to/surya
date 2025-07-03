from typing import Dict

import torch

from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.logging import configure_logging
from surya.ocr_error import OCRErrorPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor

configure_logging()


def load_predictors(
    device: str | torch.device | None = None, dtype: torch.dtype | str | None = None
) -> Dict[str, BasePredictor]:
    foundation_predictor = FoundationPredictor(device=device, dtype=dtype)
    return {
        "layout": LayoutPredictor(foundation_predictor),
        "ocr_error": OCRErrorPredictor(device=device, dtype=dtype),
        "recognition": RecognitionPredictor(foundation_predictor),
        "detection": DetectionPredictor(device=device, dtype=dtype),
        "table_rec": TableRecPredictor(device=device, dtype=dtype),
    }
