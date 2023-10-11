from .fcos import FCOS
from ..builder import DETECTORS
from ..roi_heads.cls_head import CLS_Head


@DETECTORS.register_module()
class FCOS_TCT(FCOS):
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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # BaseDetector
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        # SingleStageDetector
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        # NEW
        if self.cls_head is not None:
            cls_binary_losses = self.cls_head.forward_train(x, gt_labels)
            losses.update(cls_binary_losses)
        # END NEW
        return losses