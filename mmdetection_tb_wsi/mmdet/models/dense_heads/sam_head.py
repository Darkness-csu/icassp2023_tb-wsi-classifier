import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmcv.cnn import constant_init, trunc_normal_init
from mmdet.core import multi_apply, reduce_mean, build_bbox_coder
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from ..builder import HEADS, build_loss
from ...utils import get_root_logger
from .anchor_free_head import AnchorFreeHead
from ..sam import PromptEncoder, BboxMaskDecoder, TwoWayTransformer

INF = 1e8

@HEADS.register_module()
class SAMHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 prompt_embed_dim=256,
                 image_size=1024,
                 vit_patch_size=16,
                 num_points_perside=10,
                 point_batch=64,
                 regress_ranges=((-1,INF),),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 bbox_coder=dict(type='DistancePointBBoxCoder'),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 pretrained = None,
                 train_cfg=None,
                 test_cfg=None,):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_points_perside = num_points_perside
        self.point_batch = point_batch
        self.regress_ranges = regress_ranges
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.pretrained = pretrained
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.image_embedding_size = image_size // vit_patch_size
        self.prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.decoder=BboxMaskDecoder(
            cls_out_channels = self.cls_out_channels,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        )
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.bbox_coder = build_bbox_coder(bbox_coder)

        if self.pretrained is not None:
            self._freeze_bboxhead()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)   
        if self.pretrained is None:
            self.apply(_init_weights)
        elif isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load bboxhead from: {self.pretrained}')
            #Load SAM-B pretrained weights
            SAM_path = {
                'SAM-B':'/home/ligaojie/LungCancer/segment-anything/sam_vit_b_01ec64.pth',
                'SAM-L':'/home/ligaojie/LungCancer/segment-anything/sam_vit_l_0b3195.pth',
                'SAM-H':'/home/ligaojie/LungCancer/segment-anything/sam_vit_h_4b8939.pth'
            }
            state_dict = torch.load(SAM_path[self.pretrained])
            bboxhead_state_dict = {}
            for key,value in state_dict.items():
                if 'prompt_encoder' in key:
                    bboxhead_state_dict[key] = value
                    continue
                elif 'mask_decoder' in key:
                    new_key = key.replace('mask_decoder','decoder')
                    bboxhead_state_dict[new_key] = value
                    continue
                else:
                    continue
                    

            msg = self.load_state_dict(bboxhead_state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> bboxhead loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_bboxhead(self):#冻结prompt_encoder, mask_decoder
        #冻结prompt_encoder
        self.prompt_encoder.eval()
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
       #冻结decoder
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        #解冻decoder中的bbox部分
        unfreeze_model = [self.decoder.bbox_token, self.decoder.bbox_reg_prediction_head, 
                          self.decoder.bbox_cls_prediction_head, self.decoder.bbox_centerness_prediction_head]
        for model in unfreeze_model:
            model.train()
            for param in model.parameters():
                param.requires_grad = True


    def forward(self, feats):
        
        # feat = feat[0] #no multi-level
        # lvl_num = len(self.num_points_perside)
        # feats = [feat for _ in range(lvl_num)]

        return multi_apply(self.forward_single, feats)

    
    def forward_single(self, x):
        #x shape(bs,c,h_e,w_e)
        #按照输入进模型的图片尺寸，按照网格均匀的取B个点
        bs,c,h_e,w_e = x.shape
        all_points = self.grid()
        reg_pred = []
        cls_pred = []
        centerness_pred = []
        for (points,) in self.batch_iterator(self.point_batch, all_points):
            batch_reg_pred, batch_cls_pred, batch_centerness_pred = self.process_batch(points, x, bs)
            reg_pred.append(batch_reg_pred)
            cls_pred.append(batch_cls_pred)
            centerness_pred.append(batch_centerness_pred)
        reg_pred = torch.cat(reg_pred, dim=1)
        cls_pred = torch.cat(cls_pred, dim=1)
        centerness_pred = torch.cat(centerness_pred, dim=1)
        return cls_pred, reg_pred, centerness_pred

    def batch_iterator(self, batch_size, *args):
        # assert len(args) > 0 and all(
        # len(a) == len(args[0]) for a in args
        # ), "Batched iteration must have inputs of all the same size."
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    def process_batch(self, points, x, bs): 
        in_points = torch.as_tensor(points, device=self.decoder.get_device) #(bn,2)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device) #(bn,)
        reg_pred, cls_pred, centerness_pred = self.predict_bbox(
            x,
            in_points[:, None, :],
            in_labels[:, None],   
            multimask_output=True
        )
        reg_pred = torch.exp(reg_pred)
        return reg_pred.unflatten(0,(bs,-1)), cls_pred.unflatten(0,(bs,-1)), centerness_pred.unflatten(0,(bs,-1))

    # def grid(self):
    #     #retuen: points shape(grid_num_perside*grid_num_perside,2)#
    #     stride = self.image_size / self.num_points_perside
    #     x_range = np.floor(np.linspace(0,self.image_size,self.num_points_perside+1)[:self.num_points_perside] + (stride/2))
    #     y_range = np.floor(np.linspace(0,self.image_size,self.num_points_perside+1)[:self.num_points_perside] + (stride/2))
    #     xy = np.meshgrid(x_range, y_range)
    #     points = np.concatenate([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1)
    #     return points 

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = np.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid(self, to_torch = False, dtype=torch.float32, device=None):
        #retuen: points shape(grid_num_perside*grid_num_perside,2)#
        stride = self.image_size / self.num_points_perside
        shift_x = shift_y = (np.arange(0, self.num_points_perside) + 0.5) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        points = np.stack([shift_xx, shift_yy], axis=-1)
        if to_torch:
            assert device and dtype, 'dtype and device should be given!'
            points = torch.as_tensor(points, dtype=dtype, device=device)
        return points




    def predict_bbox(
        self,
        img_features,
        point_coords,
        point_labels,
        boxes = None,
        mask_input = None,
        multimask_output = True,
    ):
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict bboxs
        _, _, reg_pred, cls_pred, centerness_pred = self.decoder(
            image_embeddings=img_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return reg_pred, cls_pred, centerness_pred
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_meta,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 3D-tensor(bs,num_points,cls_channels)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 3D-tensor(bs,num_points,4)
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 3D-tensor(bs,num_points,1)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        lvl_num = len(cls_scores)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device
        
        #points = torch.as_tensor(self.grid(), device=device)
        points = self.grid(to_torch=True, dtype=dtype, device=device)
        all_points = [points for _ in range(lvl_num)]

        labels, bbox_targets = self.get_targets(all_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.reshape(-1) 
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_points]) #each points shape(bs*num_points,2)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        #num_pos = max(num_pos, 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        # centerness_denorm = max(
        #     pos_centerness_targets.sum().detach(), 1e-6)
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            #print(loss_bbox)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            #print(loss_bbox)
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0 #shape (num_points,num_gt)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0] #shape (num_points,num_gt)
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        # labels的size为(num_gts,),gt_labels[min_area_inds]的操作就是
        # 生成和min_area_inds的size一样的tensor,每个位置的值是索引对应的gt_label值
        # 所以labels的size为（num_points,），即为每个点的label
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        # 生成的bbox_targets的size为(num_points，4),即每个点对应的target
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        '''
            Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 3D-tensor, has shape
                (batch_size, num_points , cls_channels).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 3D-tensor, has shape
                (batch_size, num_points, 4).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 3D-tensor, has shape
                (batch_size, num_points, 1). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
            Returns:
                list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                    The first item is an (n, 5) tensor, where the first 4 columns
                    are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                    5-th column is a score between 0 and 1. The second item is a
                    (n,) tensor where each item is the predicted class label of
                    the corresponding box.
        '''
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device

        #points = torch.as_tensor(self.grid(), device=device)
        points = self.grid(to_torch=True, dtype=dtype, device=device)
        mlvl_points = [points for _ in range(num_levels)]

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]
        
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                score_factor_list, mlvl_points,
                                                img_meta, cfg, rescale, with_nms,
                                                **kwargs)
            result_list.append(results)

        return result_list
    
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_points, cls_channels).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_points, 4).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_points, 1).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
            if with_score_factors:
                score_factor = score_factor.reshape(-1).sigmoid()

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)
    
    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually with_nms is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
           mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes.to(torch.float32), mlvl_scores.to(torch.float32),
                                                mlvl_labels.to(torch.float32), cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels