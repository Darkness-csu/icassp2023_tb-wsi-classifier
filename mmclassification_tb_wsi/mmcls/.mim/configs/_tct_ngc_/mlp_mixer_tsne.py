_base_ = [
    '../_base_/models/mlp_mixer_base_tct_ngc.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]


classes = [
    'neg',
    'pos'
]
model = dict(
    backbone=dict(patch_num=200, arch='tb'),

    head=dict(
        type='LinearClsHead_TSNE'
    )

)

dataset_type = 'TCT'
data_root = '../../../commonfile/processed/TCT_NGC_DETR/smear_cls_head_query_full/'
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
    samples_per_gpu=1,
    #samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'new_train_v2',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'new_val_v2',
        ann_file=data_root + 'new_val_v2.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        classes=classes,
        data_prefix=data_root + 'new_all',
        ann_file=data_root + 'new_all.txt',
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metric=['accuracy', 'precision', 'recall', 'f1_score'],
    metric_options=dict(
        topk=(1,),  # topk=(1, 5),
        choose_classes=[1]
    )
)

#find_unused_paramters=True
work_dir = 'work_dirs/mlp_mixer_tct-ngc/cls_head_v4'
#runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=2)
checkpoint_config = dict(interval=30)