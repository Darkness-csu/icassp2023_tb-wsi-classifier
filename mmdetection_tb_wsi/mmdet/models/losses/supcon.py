import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from ..utils import concat_all_gather
import torch.distributed as dist

@LOSSES.register_module()
class SupConLoss(nn.Module):
    def __init__(self,
                 loss_weight=0.1,
                 temperature=0.1,
                 base_temperature=0.07,
                 gather=True,
                 gather_grad=False,):
        super(SupConLoss, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.gather = gather
        self.gather_grad = gather_grad


    def _contrastive(self, feature, label, gather_feats, gather_label):

        '''

        Args:
            feature: shape (N, C) 
            label: shape (N)
            gather_feats: shape (gpu_num*N, C)
            gather_label: shape (gpu_num*N)
        Returns:
            loss: shape (1)
        '''
        mini_size = feature.shape[0] 
        size = gather_feats.shape[0]
        local_mask = ~torch.eye(mini_size, dtype=torch.bool,
                                device=feature.device)  # shape (mini_batch,mini_batch)
        if dist.is_initialized() and self.gather:
            rank = torch.distributed.get_rank()
            logits_mask = torch.ones((mini_size, size), dtype=torch.bool, device=feature.device)
            logits_mask[:, rank * mini_size: (rank + 1) * mini_size] &= local_mask
        else:
            logits_mask = local_mask


        label_mask = torch.eq(label.unsqueeze(dim=1), gather_label.unsqueeze(dim=0)) # shape (mini_size, size)

        dot_contrast = torch.div(torch.einsum('ij,kj->ik',
                    feature, gather_feats), self.temperature)

        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        dot_contrast = dot_contrast - logits_max.detach()

        label_mask = torch.mul(label_mask, logits_mask)
        exp_logits = torch.exp(dot_contrast) * logits_mask

        log_prob = dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-16)
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-16)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean() * self.loss_weight
        
        return loss



    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        if dist.is_initialized() and self.gather:
            gather_feats = concat_all_gather(features, self.gather_grad)  # shape (N * num_device, C)
            gather_label = concat_all_gather(labels, self.gather_grad)  # shape (N * num_device)
            
        else:
            gather_feats = features
            gather_label = labels

        loss = self._contrastive(features, labels, 
                                 gather_feats, gather_label,)

        return loss