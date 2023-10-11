_base_ = './faster_rcnn_r50_fpn_tct_2x.py'

model = dict(
    type='FasterRCNN_TCT',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/faster_rcnn_r50_fpn_tct_2x/epoch_24.pth'
    ),
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    cls_head=dict(
        in_channels=256,
        mid_channels=10,
        in_index=0,
        loss_weight=0.3,
        pos_list=[1, 2, 3, 4, 5]
    )
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[3, ]
)
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6, metric='bbox')
