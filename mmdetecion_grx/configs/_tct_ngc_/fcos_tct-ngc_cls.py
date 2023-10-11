_base_ = './fcos-new_r50_caffe_fpn_gn-head_1x_tct-ngc.py'

model = dict(
    type='FCOS_TCT',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='new_work_dirs/fcos_tct-ngc/epoch_12.pth'
    ),
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    cls_head=dict(
        in_channels=256,
        in_index=0,
        loss_weight=0.3,
        pos_list=[1, 2, 3, 4, 5]  # 类别序号从0开始
    )
)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3
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

work_dir = 'new_work_dirs/fcos_tct-ngc_cls'
# FOR DEBUG
# runner = dict(type='IterBasedRunner', max_iters=500, max_epochs=None)
# evaluation = dict(interval=500, metric='bbox')
# checkpoint_config = dict(interval=500)
# log_config = dict(interval=50)
