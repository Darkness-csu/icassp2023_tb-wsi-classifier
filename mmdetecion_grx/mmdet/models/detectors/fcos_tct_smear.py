import torch
import torch.nn.functional as F

from mmdet.core import multi_apply
from .fcos_tct import FCOS_TCT
from ..builder import DETECTORS


@DETECTORS.register_module()
class FCOS_TCT_SMEAR(FCOS_TCT):
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
                         test_cfg, pretrained, init_cfg, cls_head)

    def simple_test(self, img, img_metas, rescale=False):
        if self.test_cfg.get('test_feature', False):
            feats = self.extract_feat(img)
            results = multi_apply(self.bbox_head_forward_single, feats)  # batch x fpn_stages x (num_classes, h, w)
            #print(results[0][4].shape)
            return self.fpn_fusion(results)
        else:
            return super().simple_test(img, img_metas, rescale=rescale)

    def bbox_head_forward_single(self, x):
        cls_feat = x

        for cls_layer in self.bbox_head.cls_convs:
            cls_feat = cls_layer(cls_feat)
        box_feat = self.bbox_head.conv_cls(cls_feat)
        return box_feat

    @staticmethod
    def fpn_fusion(results):
        outputs = []
        for result in results:
            result = [x.unsqueeze(0) for x in result]
            output = torch.zeros((1,6,100,148), device=result[0].device)
            for x in result:
                output += F.interpolate(x, output.shape[2:], mode='bilinear', align_corners=False)
                #print(output.shape)
            outputs.append([output[0]])
        return outputs

    # based on: # mmdet/models/detectors/single_stage.py
    def onnx_export(self, img, img_metas, with_nms=True):
        print(img.shape)

        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        box_feat = outs[0]
        box_feat = [F.interpolate(x, box_feat[0].shape[2:], mode='bilinear', align_corners=False) for x in box_feat]
        box_feat = torch.stack(box_feat, dim=-1)
        box_feat = torch.sum(box_feat, dim=-1)

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas, with_nms=with_nms)
        
        if self.cls_head is not None:
            cls_score = self.cls_head.onnx_export(x)
            return det_bboxes, det_labels, box_feat, cls_score
        else:
            return det_bboxes, det_labels, box_feat
