import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.bbox_embed = nn.ModuleList(
            [
                nn.Embedding(
                    config.bbox_size + config.special_token_count,
                    config.bbox_embed_size,
                )
                for _ in range(6)
            ]
        )
        self.max_bbox_embedding = config.bbox_size + config.special_token_count - 1
        self.max_bbox_size = config.bbox_size

    def embed(
        self,
        input_tokens: torch.Tensor,
        input_boxes: torch.Tensor | None,
        embed_boxes: torch.Tensor,
    ) -> torch.Tensor:
        # Embed tokens
        token_embeds = self.token_embed(input_tokens)

        # Optionally embed boxes
        if input_boxes is not None and embed_boxes.any():  # Is none in prefill
            input_boxes = input_boxes.to(torch.long)
            bbox_loss_ignore_mask = (
                (input_boxes[:, :, 0] < 0) | (input_boxes[:, :, 0] > self.max_bbox_size)
            ).unsqueeze(-1)
            input_boxes = torch.clamp(input_boxes, 0, self.max_bbox_embedding)

            bbox_embeds = torch.sum(
                torch.stack(
                    [
                        self.bbox_embed[i](input_boxes[:, :, i])
                        for i in range(len(self.bbox_embed))
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

            bbox_embeds = F.pad(
                bbox_embeds, (token_embeds.shape[-1] - bbox_embeds.shape[-1], 0)
            )
            embed_boxes = embed_boxes.unsqueeze(1).unsqueeze(1).expand_as(bbox_embeds)
            bbox_loss_ignore_mask = bbox_loss_ignore_mask.expand_as(bbox_embeds)

            mask = embed_boxes & ~bbox_loss_ignore_mask
            bbox_embeds *= mask.float()

            token_embeds = token_embeds + bbox_embeds

        return token_embeds
