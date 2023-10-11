import torch
import torch.nn as nn
from .two_stage import TwoStageDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class FS_FasterRCNN_SMEAR(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 smear_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(FS_FasterRCNN_SMEAR, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.smear_cfg = smear_cfg
        self.in_index = self.smear_cfg.get('in_index', -1)
        self.in_channels = self.smear_cfg.get('in_channels', 256)
        self.mid_channels = self.smear_cfg.get('mid_channels', 6)
        self.conv = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)


    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        if self.smear_cfg.get('test_feature', False):
            feats = self.extract_feat(img)
            feats = feats[self.in_index]
            feats = self.conv(feats)
            return [feats]
        else:
            return super().simple_test(img, img_metas, proposals=proposals, rescale=rescale)

