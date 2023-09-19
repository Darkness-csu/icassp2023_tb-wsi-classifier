# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .mask_decoder import MLP

class BboxMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        cls_out_channels: int,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        bbox_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        bbox_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.cls_out_channels = cls_out_channels

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.bbox_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        #bbox检测头，回归和分类，沿用Fcos
        self.bbox_reg_prediction_head = MLP(
            transformer_dim, bbox_head_hidden_dim, 4, bbox_head_depth
        )
        self.bbox_cls_prediction_head = MLP(
            transformer_dim, bbox_head_hidden_dim, self.cls_out_channels, bbox_head_depth
        )
        self.bbox_centerness_prediction_head = MLP(
            transformer_dim, bbox_head_hidden_dim, 1, bbox_head_depth
        )
        
    @property
    def get_device(self):
        return self.iou_token.weight.device

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, bbox_reg_pred, bbox_cls_pred, bbox_centerness_pred = self.predict_masks_bbox(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, bbox_reg_pred, bbox_cls_pred, bbox_centerness_pred

    def predict_masks_bbox(
        self,
        image_embeddings: torch.Tensor,#(bs×C×H_e×W_e)              bs和B含义不一样，B指的是point prompt的批数量
        image_pe: torch.Tensor,#(1×C×H_e×W_e)
        sparse_prompt_embeddings: torch.Tensor,#(B×N×C)         N == 1，加上padding点的话就是2
        dense_prompt_embeddings: torch.Tensor, #(B×C×H_e×W_e)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        bs = image_embeddings.shape[0]
        image_pe = image_pe.expand(bs, -1, -1, -1)  #(1×C×e_H×e_W) -> (bs×C×H×W)

        # Concatenate output tokens, (6×C) -> (B×(6+N)×C)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.bbox_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        #(B×(6+N)×C) -> (bs×B×(6+N)×C)
        tokens = tokens.unsqueeze(0).expand(bs, -1, -1 ,-1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings[:,None,...].repeat(1,tokens.shape[1], 1, 1, 1)
        src = src + dense_prompt_embeddings.unsqueeze(0).expand(bs, -1, -1, -1, -1)
        pos_src = image_pe[:,None,...].repeat(1,tokens.shape[1], 1, 1, 1)
        _, b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src.flatten(0,1), pos_src.flatten(0,1), tokens.flatten(0,1)) # hs的shape((bs*b)×6×C) src的shape((bs*b)×(h*w)×C)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        bbox_tokens_out = hs[:, (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(-1, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #shape ((bs*b)×mask_num×h_scale×w_scale)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Generate bbox reg and cls predictions
        bbox_reg_pred = self.bbox_reg_prediction_head(bbox_tokens_out)  #shape ((bs*b)×4)
        bbox_cls_pred = self.bbox_cls_prediction_head(bbox_tokens_out)  #shape ((bs*b)×num_class)
        bbox_centerness_pred = self.bbox_centerness_prediction_head(bbox_tokens_out)  #shape ((bs*b)×1)

        return masks, iou_pred, bbox_reg_pred, bbox_cls_pred, bbox_centerness_pred

