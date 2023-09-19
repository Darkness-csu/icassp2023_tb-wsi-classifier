import torch
import torch.nn as nn
import torch.nn.functional as F


# from ..builder import build_loss


class CLS_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 num_fc=0,
                 in_index=-1,
                 loss_weight=0.,
                 pos_list=[1, 2, 3, 4, 5]
                 ):
        super().__init__()
        self.pos_list = pos_list

#         self.gap = nn.AdaptiveAvgPool2d(1)

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
#         x = x_in[self.in_index]
        x = x_in.mean(axis=-1,keepdim=True)
#         print(x.size())
        x = x.view(x.size(0), -1)
#         print(x.size())
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

    def forward_test(self, x):
        #x = x[self.in_index]
        x = x.mean(axis=-1,keepdim=True)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def simple_test(self, x):
        x = self.forward_test(x)
        #return x.detach().numpy().tolist()        #make sure decoder_querys and cls_result in the same device
        return x.detach().cpu().numpy().tolist()

    def onnx_export(self, x):
        return self.forward_test(x)