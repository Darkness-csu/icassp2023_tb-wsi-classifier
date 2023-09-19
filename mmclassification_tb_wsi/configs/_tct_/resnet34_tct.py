_base_ = [
    '../_base_/default_runtime.py'
]
in_channels = 10
num_classes = 2
classes = [
    'neg',
    'pos'
]

dataset_type = 'TCT'
data_root = '../../../commonfile/processed/TCT/smear_full/'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        in_channels=in_channels,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0.25, 0.75]),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
        topk=(1,),
    ))

train_pipeline = [
    dict(type='LoadPtFromFile', self_normalize=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadPtFromFile', self_normalize=True),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'test',
        ann_file=data_root + 'test.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'test',
        ann_file=data_root + 'test.txt',
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metric=['accuracy', 'precision', 'recall', 'f1_score'],
    metric_options=dict(
        topk=(1,),  # topk=(1, 5),
        choose_classes=[1]
    )
)

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', step=[15, 30])
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 10, min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(interval=2)
checkpoint_config = dict(interval=5)
