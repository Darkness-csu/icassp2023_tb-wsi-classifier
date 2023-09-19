# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr import DETR
from ..roi_heads.deformdetr_cls_head import CLS_Head


@DETECTORS.register_module()
class DeformableDETR(DETR):

    def __init__(self, *args, **kwargs):
        
        super(DETR, self).__init__(*args, **kwargs)
