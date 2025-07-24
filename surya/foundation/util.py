from typing import List, Tuple
import numpy as np
import torch

def detect_repeat_token(predicted_tokens: List[int], max_repeats: int = 40):
    if len(predicted_tokens) < max_repeats:
        return False

    # Detect repeats containing 1 or 2 tokens
    last_n = predicted_tokens[-max_repeats:]
    unique_tokens = len(set(last_n))
    if unique_tokens > 5:
        return False

    return last_n[-unique_tokens:] == last_n[-unique_tokens * 2 : -unique_tokens]

def prediction_to_polygon_batch(
    pred: torch.Tensor,
    img_sizes: List[Tuple[int, int]],
    bbox_scaler,
    skew_scaler,
    skew_min=0.001,
):
    img_sizes = torch.from_numpy(np.array(img_sizes, dtype=np.float32)).to(
        pred.device
    )
    w_scale = (img_sizes[:, 1] / bbox_scaler)[:, None, None]
    h_scale = (img_sizes[:, 0] / bbox_scaler)[:, None, None]

    cx = pred[:, :, 0]
    cy = pred[:, :, 1]
    width = pred[:, :, 2]
    height = pred[:, :, 3]

    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2

    skew_x = torch.floor((pred[:, :, 4] - skew_scaler) / 2)
    skew_y = torch.floor((pred[:, :, 5] - skew_scaler) / 2)

    skew_x[torch.abs(skew_x) < skew_min] = 0
    skew_y[torch.abs(skew_y) < skew_min] = 0

    polygons_flat = torch.stack(
        [
            x1 - skew_x,
            y1 - skew_y,
            x2 - skew_x,
            y1 + skew_y,
            x2 + skew_x,
            y2 + skew_y,
            x1 + skew_x,
            y2 - skew_y,
        ],
        dim=2,
    )

    batch_size, seq_len, _ = pred.shape
    polygons = polygons_flat.view(batch_size, seq_len, 4, 2)

    polygons[:, :, :, 0] *= w_scale
    polygons[:, :, :, 1] *= h_scale

    return polygons