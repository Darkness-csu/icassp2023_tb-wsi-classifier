# dataset settings
# dataset_type = 'CocoDataset'
dataset_type = 'CocoDataset_TCT'
data_root = '../../../commonfile/TCTAnnotated(non-gynecologic)/'
ann_root = '../../../commonfile/tct_ngc_data/annotations/'
#ann_root = '../../../commonfile/tct_ngc_data/new protocol/annotations/'
classes = (
    #'mesothelial_cell', 
    # 'blood_cell',
    # 'tissue_cell',
    'adenocarcinoma',
    'squamous_cell_carcinoma',
    'small_cell_carcinoma',
    # 'lymphoma_rare',
    'mesothelioma_common',
    'diseased_cell',
    'NILM'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # mean=[207.680, 179.993, 203.590],
        # std=[29.683, 51.814, 43.835],
        to_rgb=True),
    #dict(type='SelfNormalize', mode=2),
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                # mean=[207.680, 179.993, 203.590],
                # std=[29.683, 51.814, 43.835],
                to_rgb=True),
            #dict(type='SelfNormalize', mode=2),
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
        ann_file=ann_root + 'train_image.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'val_image.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_root + 'test_image.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=4, metric='bbox', evaluate_per_class=True)
