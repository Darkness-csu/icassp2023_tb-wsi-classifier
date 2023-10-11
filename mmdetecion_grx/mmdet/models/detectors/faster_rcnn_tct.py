import torch

from .two_stage import TwoStageDetector
from ..builder import DETECTORS
# from ..roi_heads.cls_head import CLS_Head
from ..roi_heads.conv_cls_head import Conv_CLS_Head

@DETECTORS.register_module()
class FasterRCNN_TCT(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 cls_head=None
                 ):
        super(FasterRCNN_TCT, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.cls_head = None
        if cls_head is not None:
            self.cls_head = Conv_CLS_Head(
                in_channels=cls_head['in_channels'],
                mid_channels=cls_head['mid_channels'],
                in_index=cls_head['in_index'],
                loss_weight=cls_head['loss_weight'],
                pos_list=cls_head['pos_list'],
            )

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # NEW
        if self.cls_head is not None:
            cls_binary_losses = self.cls_head.forward_train(x, gt_labels)
            losses.update(cls_binary_losses)
        # END NEW
        return losses

    # def simple_test(self, img, img_metas, gt_labels,proposals=None, rescale=False):
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     if proposals is None:
    #         proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals
    
    #     box_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
    #     cls_labels,cls_results = self.cls_head.simple_test(x,gt_labels)
    
    #     return list(list(i) for i in zip(cls_labels, cls_results))

    # def onnx_export(self, img, img_metas):
    #     img_shape = torch._shape_as_tensor(img)[2:]
    #     img_metas[0]['img_shape_for_onnx'] = img_shape
    #     x = self.extract_feat(img)
    #     proposals = self.rpn_head.onnx_export(x, img_metas)
    #     if hasattr(self.roi_head, 'onnx_export'):
    #         det_bboxes, det_labels = self.roi_head.onnx_export(x, proposals, img_metas)
    #         if self.cls_head is None:
    #             return det_bboxes, det_labels
    #         else:
    #             image_scores = self.cls_head.onnx_export(x)
    #             return det_bboxes, det_labels, image_scores
    #     else:
    #         raise NotImplementedError(f'{self.__class__.__name__} can not be exported to ONNX')
