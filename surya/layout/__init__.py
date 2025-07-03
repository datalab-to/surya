from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.common.util import clean_boxes
from surya.layout.schema import LayoutBox, LayoutResult
from surya.settings import settings
from surya.foundation import FoundationPredictor, TaskNames
from surya.foundation.util import prediction_to_polygon_batch
from surya.input.processing import convert_if_not_rgb
from surya.layout.label import LAYOUT_PRED_RELABEL

class LayoutPredictor(BasePredictor):
    batch_size = settings.LAYOUT_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 4,
        "mps": 4,
        "cuda": 32,
        "xla": 16
    }

    # Override base init - Do not load model
    def __init__(self, foundation_predictor: FoundationPredictor):
        self.foundation_predictor = foundation_predictor
        self.processor = self.foundation_predictor.processor
        self.bbox_size = self.foundation_predictor.model.config.bbox_size
        self.tasks = self.foundation_predictor.tasks

    def __call__(
            self,
            images: List[Image.Image],
            batch_size: int | None = None,
            top_k: int = 5
    ) -> List[LayoutResult]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()

        images = convert_if_not_rgb(images)
        images = [self.processor.image_processor(image) for image in images]

        predicted_tokens, batch_bboxes, scores, topk_scores = self.foundation_predictor.prediction_loop(
            images=images,
            input_texts=["" for _ in range(len(images))],
            task_names=[TaskNames.layout for _ in range(len(images))],
            batch_size=batch_size,
            max_lookahead_tokens=0      # Do not do MTP for layout
        )
        
        image_sizes = [img.shape for img in images]
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, image_sizes, self.bbox_size, self.bbox_size // 2
        )

        layout_results = []
        for image, image_tokens, image_polygons, image_scores, image_topk_scores in zip(images, predicted_tokens, predicted_polygons, scores, topk_scores):
            layout_boxes = []
            for z, (tok, poly, score, tok_topk) in enumerate(zip(image_tokens, image_polygons, image_scores, image_topk_scores)):
                if tok == self.processor.eos_token_id:
                    break

                predicted_label = self.processor.decode([tok], "layout")
                label = LAYOUT_PRED_RELABEL[predicted_label]

                top_k_dict = {}
                for k, v in tok_topk.items():
                    l = self.processor.decode([k], "layout")
                    if l in LAYOUT_PRED_RELABEL:
                        l = LAYOUT_PRED_RELABEL[l]
                    top_k_dict.update({l: v})
                layout_boxes.append(LayoutBox(
                    polygon=poly.tolist(),
                    label=label,
                    position=z,
                    top_k=top_k_dict,
                    confidence=score
                ))
            # layout_boxes = clean_boxes(layout_boxes)
            layout_results.append(LayoutResult(
                bboxes=layout_boxes,
                image_bbox=[0, 0, *image.shape]
            ))
            

        assert len(layout_results) == len(images)
        return layout_results