from typing import List

from PIL import Image

from surya.common.predictor import BasePredictor
from surya.layout.schema import LayoutBox, LayoutResult
from surya.settings import settings
from surya.foundation import FoundationPredictor, TaskNames
from surya.foundation.util import prediction_to_polygon_batch
from surya.input.processing import convert_if_not_rgb
from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.common.util import clean_boxes


class LayoutPredictor(BasePredictor):
    batch_size = settings.LAYOUT_BATCH_SIZE
    default_batch_sizes = {"cpu": 4, "mps": 4, "cuda": 32, "xla": 16}

    # Override base init - Do not load model
    def __init__(self, foundation_predictor: FoundationPredictor):
        self.foundation_predictor = foundation_predictor
        self.processor = self.foundation_predictor.processor
        self.bbox_size = self.foundation_predictor.model.config.bbox_size
        self.tasks = self.foundation_predictor.tasks

    # Special handling for disable tqdm to pass into foundation predictor
    # Make sure they are kept in sync
    @property
    def disable_tqdm(self) -> bool:
        return super().disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)
        self.foundation_predictor.disable_tqdm = bool(value)

    def __call__(
        self, images: List[Image.Image], batch_size: int | None = None, top_k: int = 5
    ) -> List[LayoutResult]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()

        if len(images) == 0:
            return []

        images = convert_if_not_rgb(images)
        images = [self.processor.image_processor(image) for image in images]

        predicted_tokens, batch_bboxes, scores, topk_scores = (
            self.foundation_predictor.prediction_loop(
                images=images,
                input_texts=["" for _ in range(len(images))],
                task_names=[TaskNames.layout for _ in range(len(images))],
                batch_size=batch_size,
                max_lookahead_tokens=0,  # Do not do MTP for layout
                top_k=5,
                max_sliding_window=576,
                max_tokens=500,
                tqdm_desc="Recognizing Layout"
            )
        )

        image_sizes = [img.shape for img in images]
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, image_sizes, self.bbox_size, self.bbox_size // 2
        )
        layout_results = []
        for image, image_tokens, image_polygons, image_scores, image_topk_scores in zip(
            images, predicted_tokens, predicted_polygons, scores, topk_scores
        ):
            layout_boxes = []
            for z, (tok, poly, score, tok_topk) in enumerate(
                zip(image_tokens, image_polygons, image_scores, image_topk_scores)
            ):
                if tok == self.processor.eos_token_id:
                    break

                predicted_label = self.processor.decode([tok], "layout")
                label = LAYOUT_PRED_RELABEL.get(predicted_label)
                if not label:
                    # Layout can sometimes return unknown labels from other objectives
                    continue

                top_k_dict = {}
                for k, v in tok_topk.items():
                    topk_label = self.processor.decode([k], "layout")
                    if topk_label in LAYOUT_PRED_RELABEL:
                        topk_label = LAYOUT_PRED_RELABEL[topk_label]
                    if not topk_label.strip():
                        continue
                    top_k_dict.update({topk_label: v})
                layout_boxes.append(
                    LayoutBox(
                        polygon=poly.tolist(),
                        label=label,
                        position=z,
                        top_k=top_k_dict,
                        confidence=score,
                    )
                )
            layout_boxes = clean_boxes(layout_boxes)
            layout_results.append(
                LayoutResult(
                    bboxes=layout_boxes,
                    image_bbox=[0, 0, image.shape[1], image.shape[0]],
                )  # Image is numpy array
            )

        assert len(layout_results) == len(images)
        return layout_results
