from typing import Optional
import torch
import torch.nn.functional as F

from surya.common.load import ModelLoader
from surya.settings import settings


class BasePredictor:
    model_loader_cls = ModelLoader
    batch_size: Optional[int] = None
    default_batch_sizes = {"cpu": 1, "mps": 1, "cuda": 1}
    torch_dtype = settings.MODEL_DTYPE

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: torch.device | str | None = settings.TORCH_DEVICE_MODEL,
        dtype: Optional[torch.dtype | str] = None,
        attention_implementation: Optional[str] = None,
    ):
        if dtype is None:
            dtype = self.torch_dtype

        self.model = None
        self.processor = None
        loader = self.model_loader_cls(checkpoint)

        self.model = loader.model(device, dtype, attention_implementation)
        self.processor = loader.processor()

        self._disable_tqdm = settings.DISABLE_TQDM

    def to(self, device_dtype: torch.device | str | None = None):
        model_moved = False
        if hasattr(self, "model") and self.model:
            self.model.to(device_dtype)
            model_moved = True
        if hasattr(self, "foundation_predictor") and self.foundation_predictor:
            self.foundation_predictor.model.to(device_dtype)
            model_moved = True

        if not model_moved:
            raise ValueError("Model not loaded")

    def get_batch_size(self):
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self.default_batch_sizes["cpu"]
            if settings.TORCH_DEVICE_MODEL in self.default_batch_sizes:
                batch_size = self.default_batch_sizes[settings.TORCH_DEVICE_MODEL]
        return batch_size

    @staticmethod
    def pad_to_batch_size(tensor: torch.Tensor, batch_size: int):
        current_batch_size = tensor.shape[0]
        if current_batch_size >= batch_size:
            return tensor

        if len(tensor.shape) == 1:
            # If tensor is 1D, we need to pad it to the batch size
            pad_size = batch_size - current_batch_size
            return F.pad(tensor, (0, pad_size), mode="constant", value=0)

        pad_size = batch_size - current_batch_size
        padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

        return F.pad(tensor, padding, mode="constant", value=0)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
