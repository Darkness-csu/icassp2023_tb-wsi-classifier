_base_ = [
    '../_base_/models/mlp_mixer_base_tct_ngc.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]


classes = [
    'neg',
    'pos'
]
model = dict(backbone=dict(patch_num=200, arch='tb'))

dataset_type = 'TCT'
data_root = '../../../commonfile/processed/TCT_NGC_DETR_961/smear_cls_head_v4_full/'
train_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    #samples_per_gpu=4,
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
        data_prefix=data_root + 'val',
        ann_file=data_root + 'val.txt',
        pipeline=test_pipeline))

evaluation = dict(
    interval=10,
    metric=['accuracy', 'precision', 'recall', 'f1_score'],
    metric_options=dict(
        topk=(1,),  # topk=(1, 5),
        choose_classes=[1]
    )
)

#find_unused_paramters=True
work_dir = 'new_work_dirs/mlp_mixer_tct-ngc/cls_head_v4'
#runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=2)
checkpoint_config = dict(interval=30)