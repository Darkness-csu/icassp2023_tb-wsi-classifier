_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_ngc.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
num_classes = 6
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)
model = dict(
    type='FasterRCNN',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/ligaojie/LungCancer/mmdetection-tct/configs/param-mmdet/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=None),
    rpn_head=dict(
        type='RPNHead',
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))),
)
evaluation = dict(interval=6, metric='bbox')

work_dir = 'new_work_dirs/doublehead_faster_tct-ngc/new_protocol'
