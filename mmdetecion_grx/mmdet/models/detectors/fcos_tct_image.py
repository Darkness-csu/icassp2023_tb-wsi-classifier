from .fcos import FCOS
from ..builder import DETECTORS
from ..roi_heads.cls_head import CLS_Head


@DETECTORS.register_module()
class FCOS_TCT_IMAGE(FCOS):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 cls_head=None
                 ):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained, init_cfg)

        self.cls_head = None
        if cls_head is not None:
            self.cls_head = CLS_Head(
                in_channels=cls_head['in_channels'],
                in_index=cls_head['in_index'],
                loss_weight=cls_head['loss_weight'],
                pos_list=cls_head['pos_list'],
            )

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        if self.test_cfg.get('cls_result', False):
            x = self.extract_feat(img)
            cls_results = self.cls_head.simple_test(x)
            return cls_results
        else:
            return super().simple_test(img, img_metas, proposals=proposals, rescale=rescale)