_base_ = [
    '../_base_/default_runtime.py'
]
in_channels = 6 
num_classes = 2
classes = [
    'neg',
    'pos'
]

dataset_type = 'TCT'
data_root = '../../../commonfile/processed/TCT_NGC/smear_full/'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MLP_TCT',
        in_channels=in_channels,
        input_num = 200*296
        ),
    head=dict(
        type='MLPHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)       
    )
   
)

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
        data_prefix=data_root + 'val',
        ann_file=data_root + 'val.txt',
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
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=10, warmup_ratio=1.0 / 10, min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=300)

work_dir = 'work_dirs/MLP_tct-ngc/test'

log_config = dict(interval=2)
checkpoint_config = dict(interval=20)
