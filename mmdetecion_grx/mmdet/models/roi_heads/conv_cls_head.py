from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_CLS_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 in_index=-1,
                 loss_weight=0.,
                 pos_list=[1, 2, 3, 4, 5]
                 ):
        super().__init__()
        self.pos_list = pos_list

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(mid_channels, 1)
        )

        self.in_index = in_index
        self.loss_weight = loss_weight

    def process_labels(self, gt_labels, device):
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
        x = x_in[self.in_index]
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        losses = dict()
        if gt_labels is None:
            losses['loss_binary_cls'] = torch.tensor(0.0).to(x.device)
        else:
            labels = self.process_labels(gt_labels, x.device)
            loss = F.binary_cross_entropy_with_logits(x, labels.float())
            losses['loss_binary_cls'] = self.loss_weight * loss
        return losses

    def forward_test(self, x):
        x = x[self.in_index]
        feat = self.conv(x)
        x = self.gap(feat)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x, feat

    def smear_test(self, x):
        x = x[self.in_index]
        feat = self.conv(x)
        return [feat]  # this format is the same as FPN features outputs

    def simple_test(self, x):
        x, _ = self.forward_test(x)
        return x.detach().cpu().numpy().tolist()


    def onnx_export(self, x):
        return self.forward_test(x)
