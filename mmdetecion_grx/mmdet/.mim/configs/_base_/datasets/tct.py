# dataset settings
# dataset_type = 'CocoDataset'
dataset_type = 'CocoDataset_TCT'
data_root = '../../../commonfile/TCTAnnotatedData/'
ann_root = '../data/TCT/annotations/'

classes = ('normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em',
           'vaginalis', 'monilia', 'dysbacteriosis_herpes_act', 'ec')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='SelfNormalize'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='SelfNormalize'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
