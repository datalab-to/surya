from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.onnx import OnnxConfig
except ImportError:
    # transformers >= 5.0 removed the built-in ONNX exporter (the
    # transformers.onnx package; export now lives in the `optimum`
    # package). DistilBertOnnxConfig below is an ONNX-export helper that
    # surya's torch inference path never uses, so when the exporter is
    # absent we skip defining it rather than fail the whole import.
    OnnxConfig = None

from surya.common.s3 import S3DownloaderMixin

ID2LABEL = {
    0: 'good',
    1: 'bad'
}

class DistilBertConfig(S3DownloaderMixin, PretrainedConfig):
    model_type = "distilbert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        super().__init__(**kwargs, pad_token_id=pad_token_id)


if OnnxConfig is not None:

    class DistilBertOnnxConfig(OnnxConfig):
        @property
        def inputs(self) -> Mapping[str, Mapping[int, str]]:
            if self.task == "multiple-choice":
                dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
            else:
                dynamic_axis = {0: "batch", 1: "sequence"}
            return OrderedDict(
                [
                    ("input_ids", dynamic_axis),
                    ("attention_mask", dynamic_axis),
                ]
            )