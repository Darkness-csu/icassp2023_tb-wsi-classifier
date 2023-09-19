import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import MultiheadAttention

#from ..utils import MultiheadAttention

# from ..builder import build_loss

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
            #qkv_bias=qkv_bias
            )

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x

class CLS_Head_V3(nn.Module): #包含了单层的vit
    def __init__(self,
                 in_channels,
                 num_fc=0,
                 in_index=-1,
                 loss_weight=0.,
                 pos_list=[1, 2, 3, 4, 5],
                 transformer_settings=dict(
                    embed_dims=256,
                    num_layers=1,
                    num_heads=4,
                    feedforward_channels=256*4,
                    drop_rate=0.,
                    drop_path_rate=0.,
                    norm_cfg=dict(type='LN', eps=1e-6),
                    qkv_bias=True
                 )
                ):
        super().__init__()
        self.pos_list = pos_list
        self.transformer_settings = transformer_settings
        dpr = np.linspace(0, transformer_settings['drop_path_rate'], self.transformer_settings['num_layers'])
        layer_cfg = dict(
                embed_dims=self.transformer_settings['embed_dims'],
                num_heads=self.transformer_settings['num_heads'],
                feedforward_channels=self.transformer_settings['feedforward_channels'],
                drop_rate=self.transformer_settings['drop_rate'],
                drop_path_rate=dpr[0],
                qkv_bias=self.transformer_settings.get('qkv_bias', True),
                norm_cfg=self.transformer_settings['norm_cfg'])

        self.single_transformer_layer = TransformerEncoderLayer(**layer_cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_settings['embed_dims']))
        #torch.nn.init.xavier_uniform_(self.cls_token)
        self.drop_after_pos = nn.Dropout(p=self.transformer_settings['drop_rate'])
        self.final_norm_layer = build_norm_layer(self.transformer_settings['norm_cfg'], self.transformer_settings['embed_dims'], postfix=1)[1]
        
        fc = []
        for i in range(num_fc):
            fc.append(nn.Linear(in_channels, in_channels))
        fc.append(nn.Linear(in_channels, 1))
        self.fc = nn.Sequential(*fc)
        # self.fc = nn.Linear(in_channels, 1)

        self.in_index = in_index
        self.loss_weight = loss_weight
        # self.loss_cls_binary = build_loss(loss_cls_binary)

    def process_labels(self, gt_labels, device):
        #print(gt_labels)
        labels = []
        for gt in gt_labels:
            label = 0.  # negative
            for index in self.pos_list:
                if index in gt:
                    label = 1.  # positive
                    break
            labels.append(label)
        return torch.tensor(labels).reshape(-1, 1).to(device)

    def forward_train(self, x_in, gt_labels=None):
        #x_in.size = [B,query_num,256]
        B = x_in.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_in = torch.cat((cls_tokens, x_in), dim=1)
        x_in = self.drop_after_pos(x_in)
        x_in = self.single_transformer_layer(x_in)
        x_in = self.final_norm_layer(x_in)
        out_cls_token = x_in[:,0]
        x = out_cls_token.view(out_cls_token.size(0), -1)
#        print(x.size())
        x = self.fc(x)

        losses = dict()
        if gt_labels is None:
            losses['loss_binary_cls'] = torch.tensor(0.0).to(x.device)
        else:
            labels = self.process_labels(gt_labels, x.device)
            #print(labels)
            loss = F.binary_cross_entropy_with_logits(x, labels.float())
            losses['loss_binary_cls'] = self.loss_weight * loss
        return losses

    def forward_test(self, x_in):
        #print(x_in.size()) # (1,300,256)
        #print(x_in) 输入不一样
        B = x_in.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #print(cls_tokens.size())
        x_in = torch.cat((cls_tokens, x_in), dim=1)
        #x_in = self.drop_after_pos(x_in)
        x_in = self.single_transformer_layer(x_in)
        print(x_in)
        x_in = self.final_norm_layer(x_in)
        #print(x_in)
        out_cls_token = x_in[:,0]
        x = out_cls_token.view(out_cls_token.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.detach().cpu().numpy().tolist()
        cls_results = torch.Tensor(x)
        #print(cls_results)
        out = torch.cat((out_cls_token.detach().cpu(), cls_results),dim=1)
        #print(out.size())
        return out

    def simple_test(self, x):
        x = self.forward_test(x)
        #return x.detach().numpy().tolist()        #make sure decoder_querys and cls_result in the same device
        return x

    def onnx_export(self, x):
        return self.forward_test(x)