_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/tct.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
num_classes = 10
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3
)
model = dict(
    type='FasterRCNN',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='../param-mmdet/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
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
        type='StandardRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
)
evaluation = dict(interval=6, metric='bbox')
