import torch

from .faster_rcnn_tct import FasterRCNN_TCT
from ..builder import DETECTORS


@DETECTORS.register_module()
class FasterRCNN_TCT_CLS(FasterRCNN_TCT):
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
        assert cls_head is not None or "cls head is None in FasterRCNN_TCT_SMEAR"
        super(FasterRCNN_TCT_CLS, self).__init__(
            backbone, rpn_head, roi_head,
            train_cfg, test_cfg, neck, pretrained, init_cfg, cls_head)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        if self.test_cfg.get('test_cls', False):
            x = self.extract_feat(img)
            cls_results = self.cls_head.simple_test(x)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            box_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
            cls_results = self.cls_head.simple_test(x)
            
            return list(list(i) for i in zip(box_results, cls_results))
            #return list(cls_results)
        else:
            return super().simple_test(img, img_metas, proposals=proposals, rescale=rescale)

    # based on: # mmdet/models/detectors/two_stage.py
    def onnx_export(self, img, img_metas):
        if not hasattr(self.roi_head, 'onnx_export'):
            raise NotImplementedError(f'{self.__class__.__name__} can not be exported to ONNX.')

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)

        det_bboxes, det_labels = self.roi_head.onnx_export(x, proposals, img_metas)
        if self.cls_head is None:
            return det_bboxes, det_labels
        else:
            cls_score, smear_feat = self.cls_head.onnx_export(x)
            return det_bboxes, det_labels, smear_feat, cls_score
