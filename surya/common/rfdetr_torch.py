"""RfDetrTorch — rf-detr (Roboflow) detector via the vendored model copy (no rfdetr package).

Backs ``fast_layout``. Inference goes through the slimmed, detection-only model
definition vendored under ``surya.common.rfdetr`` (validated byte-for-byte against the
upstream rfdetr package). Pure PyTorch — runs on cpu/mps/cuda.

Model dir layout (downloaded from the Hub):
  rfdetr_<task>.pth   the fine-tuned rf-detr weights
  config.json         {"arch": "rf-detr-large", "categories": [{"id", "name"}, ...], ...}
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

from PIL import Image


class _DetList(list):
    """A list of detections that can also carry the page's encoder feature map (.features)."""

    features = None


_MPS_OP_OK: Optional[bool] = None


def _mps_op_supported() -> bool:
    """Probe (once) whether the DINOv2 pos-embed op the rf-detr backbone needs runs on
    MPS. It uses antialiased bicubic (aten::_upsample_bicubic2d_aa), which has no MPS
    kernel — it only works if PYTORCH_ENABLE_MPS_FALLBACK=1 was set before `import torch`
    (surya/__init__.py sets it, but that's too late if torch was already imported).
    Probing at runtime is the only reliable signal, since the env var reads "1" even when
    torch ignored it."""
    global _MPS_OP_OK
    if _MPS_OP_OK is None:
        import warnings

        import torch

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.nn.functional.interpolate(
                    torch.zeros(1, 1, 4, 4, device="mps"),
                    size=(6, 6),
                    mode="bicubic",
                    antialias=True,
                )
            _MPS_OP_OK = True
        except Exception:
            _MPS_OP_OK = False
    return _MPS_OP_OK


def _pick_device(device: Optional[str]) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    # Use MPS only if the CPU-fallback for the unsupported bicubic op is actually live
    # (see _mps_op_supported); otherwise auto-select CPU (rf-detr is ~0.3s/page on CPU)
    # so Apple Silicon never crashes regardless of import order.
    if torch.backends.mps.is_available() and _mps_op_supported():
        return "mps"
    return "cpu"


class RfDetrTorch:
    def __init__(
        self,
        model_dir: str,
        num_threads: Optional[int] = None,
        device: Optional[str] = None,
    ):
        import torch

        if num_threads:
            torch.set_num_threads(int(num_threads))

        self.device = _pick_device(device)
        if self.device == "mps" and not _mps_op_supported():
            # Only reached when the user explicitly forces mps (auto-select already
            # probes and drops to CPU — see _pick_device). The DINOv2 pos-embed op has
            # no MPS kernel and will raise unless PYTORCH_ENABLE_MPS_FALLBACK=1 was set
            # BEFORE torch was imported (surya/__init__.py sets it, but that's too late
            # if torch was already loaded).
            from surya.logging import get_logger

            get_logger().warning(
                "FAST_DETECTOR_DEVICE=mps but the MPS bicubic fallback isn't active "
                "(torch was likely imported before surya); the rf-detr detector will "
                "crash. Export PYTORCH_ENABLE_MPS_FALLBACK=1 before importing torch, or use cpu/cuda."
            )

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        # rf-detr's predict() returns 0-indexed class ids that line up with the COCO
        # categories sorted by id (Row=0, Col=1; layout: Caption=0 ... Text=15).
        cats = sorted(cfg["categories"], key=lambda c: c["id"])
        self.id2label = {i: c["name"] for i, c in enumerate(cats)}

        weights = cfg.get("weights")
        weights = os.path.join(model_dir, weights) if weights else None
        if not weights or not os.path.exists(weights):
            pths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
            if not pths:
                raise FileNotFoundError(f"no rf-detr .pth weights found in {model_dir}")
            weights = pths[0]

        arch = (cfg.get("arch") or "rf-detr-large").lower()
        if "base" in arch:
            raise ValueError(
                "vendored rf-detr copy is rf-detr-large only; got arch=%r" % arch
            )
        from surya.common.rfdetr import RFDetrDetector

        # Honor resolution / PE overrides from config.json so reduced-resolution
        # fine-tunes (e.g. the 448 layout model) run at their trained size; absent
        # these keys the predictor falls back to LARGE_ARGS (704), so older configs
        # are unaffected.
        arch_args = {}
        if cfg.get("resolution"):
            arch_args["resolution"] = int(cfg["resolution"])
        if cfg.get("positional_encoding_size"):
            arch_args["positional_encoding_size"] = int(cfg["positional_encoding_size"])
        self.model = RFDetrDetector(
            weights_path=weights, device=self.device, arch_args=arch_args or None
        )

    def detect(
        self,
        images: List[Image.Image],
        threshold: float = 0.4,
        batch_size: int = 8,
        return_features: bool = False,
    ) -> List[List[dict]]:
        """Returns, per image, a list of {label, label_id, score, bbox:[x0,y0,x1,y1] pixels}.
        When return_features=True, each per-image list carries the encoder feature map on a
        ``.features`` attribute ([C,F,F] tensor) for the reading-order head."""
        out: List = []
        for s in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[s : s + batch_size]]
            for det in self.model.predict(
                chunk, threshold=threshold, return_features=return_features
            ):
                boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
                dets: List[dict] = _DetList()
                for i in range(len(scores)):
                    cid = int(labels[i])
                    x0, y0, x1, y1 = (float(v) for v in boxes[i].tolist())
                    dets.append(
                        {
                            "label": self.id2label.get(cid, str(cid)),
                            "label_id": cid,
                            "score": float(scores[i]),
                            "bbox": [x0, y0, x1, y1],
                        }
                    )
                if return_features:
                    dets.features = det.get("features")
                out.append(dets)
        return out


def load_detector(
    model_dir: str, num_threads: Optional[int] = None, device: Optional[str] = None
):
    """Build the rf-detr torch detector from a model dir containing ``.pth`` weights
    + ``config.json``. On CPU the torch rf-detr runs ~0.3s/page, so it remains the
    fast-mode path."""
    return RfDetrTorch(model_dir, num_threads=num_threads, device=device)


def resolve_model_dir(checkpoint: str) -> str:
    """Resolve a fast-model checkpoint to a local dir. Supports a plain local path, an
    ``hf://<repo>/<subfolder>`` ref (downloaded from the Hub), or an ``s3://`` path."""
    if checkpoint and checkpoint.startswith("hf://"):
        from huggingface_hub import snapshot_download

        parts = checkpoint[len("hf://") :].split("/")
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        local = snapshot_download(
            repo_id,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        )
        return os.path.join(local, subfolder) if subfolder else local
    if checkpoint and os.path.isdir(checkpoint):
        return checkpoint
    if checkpoint and checkpoint.startswith("s3://"):
        from surya.common.s3 import (  # type: ignore
            S3DownloaderMixin,
            download_directory,
        )

        local_path = S3DownloaderMixin.get_local_path(checkpoint)
        remote = checkpoint.replace(S3DownloaderMixin.s3_prefix, "")
        download_directory(remote, local_path)
        return local_path
    raise FileNotFoundError(
        f"fast-model checkpoint not found as a local dir: {checkpoint!r}"
    )
