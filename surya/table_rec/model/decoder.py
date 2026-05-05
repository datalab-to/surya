from typing import Optional, Tuple, Union

import torch
from torch import nn

from surya.common.adetr.decoder import SuryaADETRDecoderModel, SuryaADETRDecoderPreTrainedModel
from surya.table_rec.model.config import TableRecModelOutput
from surya.table_rec.shaper import LabelShaper
from surya.settings import settings


class LabelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Bboxes
        self.w_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.h_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.cx_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.cy_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.xskew_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.yskew_embed = nn.Embedding(config.vocab_size, config.box_embed_size)

        self.x1_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.y1_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.x2_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.y2_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.x3_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.y3_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.x4_embed = nn.Embedding(config.vocab_size, config.box_embed_size)
        self.y4_embed = nn.Embedding(config.vocab_size, config.box_embed_size)

        # Get indexes for passed in tensor
        shaper = LabelShaper()
        self.component_idxs = shaper.component_idx_dict()
        merge_count = shaper.get_box_property("merges")[1] + config.special_token_count
        category_count = shaper.get_box_property("category")[1] + config.special_token_count

        # Other box properties
        self.category_embed = nn.Embedding(category_count, config.property_embed_size)
        self.merge_embed = nn.Embedding(merge_count, config.property_embed_size)
        self.colspan_embed = nn.Embedding(config.vocab_size, config.property_embed_size)

        self.config = config

    def forward(self, boxes: torch.LongTensor, *args):
        # Need to keep *args for compatibility with common decoder
        boxes = boxes.to(torch.long).clamp(0, self.config.vocab_size)

        boxes_unbound = boxes.to(torch.long).unbind(dim=-1)
        cx, cy, w, h, xskew, yskew = boxes_unbound[self.component_idxs["bbox"][0]:self.component_idxs["bbox"][1]]
        category = boxes_unbound[self.component_idxs["category"][0]:self.component_idxs["category"][1]][0]
        merges = boxes_unbound[self.component_idxs["merges"][0]:self.component_idxs["merges"][1]][0]
        colspan = boxes_unbound[self.component_idxs["colspan"][0]:self.component_idxs["colspan"][1]][0]

        xskew_actual = ((xskew - self.config.bbox_size // 2) / 2).to(torch.long)
        yskew_actual = ((yskew - self.config.bbox_size // 2) / 2).to(torch.long)

        x1 = (cx - w // 2 - xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y1 = (cy - h // 2 - yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        x3 = (cx + w // 2 + xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y3 = (cy + h // 2 + yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)

        size_embeds = self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)
        skew_embeds = self.xskew_embed(xskew) + self.yskew_embed(yskew)
        corner_embeds = self.x1_embed(x1) + self.y1_embed(y1) + self.x3_embed(x3) + self.y3_embed(y3)
        box_embeds = size_embeds + skew_embeds + corner_embeds

        property_embeds = self.category_embed(category) + self.merge_embed(merges) + self.colspan_embed(colspan)

        # Cat bbox and property embeddings
        embedded = torch.cat([box_embeds, property_embeds], dim=-1)
        return embedded


class SuryaTableRecDecoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None
    all_tied_weights_keys = {}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = LabelEmbedding(config)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.TABLE_REC_STATIC_CACHE,
            max_boxes=settings.TABLE_REC_MAX_BOXES
        )
        self.vocab_size = config.vocab_size

        shaper = LabelShaper()
        property_heads = {}
        for k in shaper.property_keys:
            _, kcount, mode = shaper.get_box_property(k)
            property_heads[k] = nn.Linear(config.hidden_size, kcount, bias=False)

        self.box_property_heads = nn.ModuleDict(property_heads)
        self.pre_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        prefill: bool = False,
        **kwargs
    ) -> Union[Tuple, TableRecModelOutput]:
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            prefill=prefill,
        )

        hidden_states = self.pre_output_norm(outputs[0])
        box_property_logits = {}
        for key in self.box_property_heads:
            box_property_logits[key] = self.box_property_heads[key](hidden_states)

        bbox_logits = nn.functional.sigmoid(box_property_logits["bbox"])
        box_property_logits["bbox"] = bbox_logits

        return TableRecModelOutput(
            box_property_logits=box_property_logits,
            hidden_states=hidden_states,
        )