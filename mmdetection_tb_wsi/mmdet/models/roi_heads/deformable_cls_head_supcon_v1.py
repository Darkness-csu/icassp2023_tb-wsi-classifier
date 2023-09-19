import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_loss

# from ..builder import build_loss

class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

class Mixer_struc(nn.Module):
    def __init__(self, patches: int , token_dim: int, dim: int,channel_dim: int,dropout = 0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(patches,token_dim,self.dropout)
        self.MLP_block_chan = MLPBlock(dim,channel_dim,self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self,x):
        out = self.LayerNorm(x)
        out = out.permute(0,2,1)
        out = self.MLP_block_token(out)
        out = out.permute(0,2,1)
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2+=out
        return out2

class MLP_Mixer(nn.Module):
    def __init__(self, patch_num, token_dim, channel_dim, dim, num_blocks):
        super(MLP_Mixer, self).__init__()
        n_patches =patch_num
        
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim,channel_dim=channel_dim,dim=dim) for i in range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        out = self.Layernorm1(x)
        out = out.mean(dim = 1) # out（n_sample,dim）
        result = self.classifier(out)
        return out,result


class CLS_Head_SupCon_V1(nn.Module): #包含小型的MLP Mixer
    def __init__(self,
                 loss_supcon,
                 in_index=-1,
                 loss_weight=0.,
                 pos_list=[1, 2, 3, 4, 5]
                 ):
        super().__init__()
        self.pos_list = pos_list
        self.in_index = in_index
        self.loss_weight = loss_weight
        self.loss_supcon = build_loss(loss_supcon)
        # self.loss_cls_binary = build_loss(loss_cls_binary)
        #self.mlp_mixer = MLP_Mixer(patch_num=300, dim=256, num_blocks=4, token_dim=128, channel_dim=1024)
        self.mlp_mixer = MLP_Mixer(patch_num=300, dim=256, num_blocks=6, token_dim=128, channel_dim=1024)
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
#         x = x_in[self.in_index]
        feat,x = self.mlp_mixer(x_in)
        losses = dict()
        if gt_labels is None:
            losses['loss_binary_cls'] = torch.tensor(0.0).to(x.device)
            losses['loss_supcon'] = torch.tensor(0.0).to(x.device)
        else:
            labels = self.process_labels(gt_labels, x.device)
            #print(labels)
            loss_cls = F.binary_cross_entropy_with_logits(x, labels.float())
            loss_supcon = self.loss_supcon(feat,labels)
            losses['loss_binary_cls'] = self.loss_weight * loss_cls
            losses['loss_supcon'] = loss_supcon
        return losses

    def forward_test(self, x_in):
        out_feature,out_logit = self.mlp_mixer(x_in)
        out_score = torch.sigmoid(out_logit)
        out_score = out_score.cpu().numpy().tolist()
        out_score = torch.Tensor(out_score)
        out = torch.cat((out_feature.cpu(), out_score),dim=1)
        #print(out.size())
        return out

    def simple_test(self, x):
        x = self.forward_test(x)
        #return x.detach().numpy().tolist()        #make sure decoder_querys and cls_result in the same device
        return x

    def onnx_export(self, x):
        return self.forward_test(x)