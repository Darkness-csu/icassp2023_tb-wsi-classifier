import torch
import torch.nn.functional as F
from ..builder import DETECTORS
from .detr import DETR
from ..roi_heads.deformdetr_cls_head import CLS_Head
from .single_stage import SingleStageDetector
#from .deformable_detr_tct import DeformableDETR_TCT
@DETECTORS.register_module()
class DeformableDETR_TCT_SMEAR_QUERY(DETR):
    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
        #self.test_cfg = test_cfg

        self.cls_head = None
        cls_head=dict(
            in_channels=256,
            in_index=0,
            loss_weight=0.3,
            pos_list=[1, 2, 3, 4, 5]
        )
        self.cls_head = CLS_Head(
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
        decoder_querys = decoder_querys.permute(1,2,0)
#         print(decoder_querys.size())
        
        if self.cls_head is not None:
            cls_binary_losses = self.cls_head.forward_train(decoder_querys, gt_labels)
            losses.update(cls_binary_losses)
        # END NEW
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        if self.test_cfg.get('test_feature', False):    
            x = self.extract_feat(img)
            
            #get the last layer decoder output
            decoder_querys = self.get_decoder_out(x,img_metas)[-1] #torch.size([300,1,256])
            decoder_querys = decoder_querys.permute(1,0,2)
            decoder_querys = decoder_querys.detach().cpu()
            # pre_decoder_querys = pre_decoder_querys.permute(1,2,0)
            # decoder_querys = pre_decoder_querys.mean(axis=-1,keepdim=True)
            # decoder_querys = decoder_querys.view(decoder_querys.size(0), -1)
            # print(decoder_querys.size()) torch.size([1,256])
            # decoder_querys = decoder_querys.detach().cpu()
            # cls_results = self.cls_head.simple_test(pre_decoder_querys)
            # cls_results = torch.Tensor(cls_results)
            
            det_result = self.bbox_head.simple_test_bboxes(x,img_metas,rescale=rescale)

            
            #print(len(det_result)) #Batchsize
            #print(len(det_result[0][0])) #test_cfg中的max_per_img
            #print(len(det_result[0][1])) #test_cfg中的max_per_img
            #print(det_result[0][0][:50]) #det_result[0][0]表示的是bbox_pred的输出结果,[1.1431e+03, 8.6510e+02, 1.2748e+03, 9.8520e+02, 5.6107e-01],最后一项为分类分数
            #print(det_result[0][1][:50]) #det_result[0][1]表示的是cls_label的输出结果，6类，（0，1，2，3，4，5），第6类是阴性
            #print(det_result[0][2][:10]) #det_result[0][2]表示的是query_index的输出结果，结果从0-299
            selected_querys = None
            bs = len(det_result)
            dim_num = decoder_querys.shape[-1]
            #thrs = 0.7
            for i in range(bs):
                bbox_preds = det_result[i][0]
                cls_labels = det_result[i][1]
                query_indexes = det_result[i][2]
                num = len(bbox_preds) #100
                for j in range(num):
                    if cls_labels[j] != 5:
                        cls_score = torch.Tensor([bbox_preds[j][-1]]).unsqueeze(0)
                        cls_label = torch.Tensor([cls_labels[j]]).unsqueeze(0)
                        #print(cls_scores.size()) [1,1]
                        if selected_querys == None:
                            new_query = decoder_querys[i][query_indexes[j]].unsqueeze(0)
                            #print(new_query.size()) [1,256]
                            selected_querys = torch.cat((new_query, cls_score, cls_label),dim=1) #size [1,258] 后面两维是cls_score和cls_label
                        else:
                            new_query = torch.cat((decoder_querys[i][query_indexes[j]].unsqueeze(0), cls_score, cls_label),dim=1)
                            selected_querys = torch.cat((selected_querys,new_query),dim=0)
            if selected_querys != None:
                selected_querys = selected_querys.reshape((bs,-1,dim_num+2))
                return selected_querys
            else:
                return torch.zeros([bs,0,dim_num+2])
            
            
        else:
            return super().simple_test(img, img_metas, proposals=proposals, rescale=rescale)