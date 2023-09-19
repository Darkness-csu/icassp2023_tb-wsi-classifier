import torch
import torch.nn.functional as F
from ..builder import DETECTORS
from .detr import DETR
from ..roi_heads.deformable_cls_head_v4 import CLS_Head_V4
from .single_stage import SingleStageDetector
#from .deformable_detr_tct import DeformableDETR_TCT
@DETECTORS.register_module()
class DeformableDETR_TCT_V4(DETR):
    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
        #self.test_cfg = test_cfg

        self.cls_head = None
        cls_head=dict(
            in_channels=256,
            in_index=0,
            loss_weight=3,#0.3
            pos_list=[1, 2, 3, 4, 5]
        )
        self.cls_head = CLS_Head_V4(
            in_channels=cls_head['in_channels'],
            in_index=cls_head['in_index'],
            loss_weight=cls_head['loss_weight'],
            pos_list=cls_head['pos_list'],
        )
            
    
    def get_decoder_out(self,mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.bbox_head.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.bbox_head.as_two_stage:
            query_embeds = self.bbox_head.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.bbox_head.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.bbox_head.reg_branches if self.bbox_head.with_box_refine else None,  # noqa:E501
                    cls_branches=self.bbox_head.cls_branches if self.bbox_head.as_two_stage else None  # noqa:E501
            )
        return hs
    
    
    
    
    
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
        # need to get decoder output
        decoder_querys = self.get_decoder_out(x,img_metas)[-1] #get the last layer decoder output
        decoder_querys = decoder_querys.permute(1,0,2)
        decoder_querys = decoder_querys.detach()
        #decoder_querys = decoder_querys.permute(1,2,0)
#         print(decoder_querys.size())
        
        if self.cls_head is not None:
            cls_binary_losses = self.cls_head.forward_train(decoder_querys, gt_labels)
            losses.update(cls_binary_losses)
            
        # END NEW
        return losses

    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     if self.test_cfg.get('test_feature', False):    
    #         x = self.extract_feat(img)
            
    #         #get the last layer decoder output
    #         pre_decoder_querys = self.get_decoder_out(x,img_metas)[-1] 
    #         #print(pre_decoder_querys.size()) torch.size([1,300,256] 
    #         decoder_querys = pre_decoder_querys.detach()
    #         out = self.cls_head.simple_test(decoder_querys)
            
    #         if self.test_cfg.get('cls_result', False):
    #             return out
    #         else:
    #             return decoder_querys
    #         #return decoder_querys
            
    #     else:
    #         return super().simple_test(img, img_metas, proposals=proposals, rescale=rescale)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        #set_requires_grad()
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs