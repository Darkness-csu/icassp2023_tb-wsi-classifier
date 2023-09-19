num_classes = 6

model = dict(
    type = 'SAM_FCOS',
    backbone=dict(
        type='ImageEncoderViT',
        depth=12,#12 24 32
        embed_dim=768,#768 1024 1280
        img_size=1024,
        mlp_ratio=4,
        num_heads=12,#12 16 16
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2,5,8,11] ,#[2,5,8,11] [5,11,17,23] [7, 15, 23, 31]
        window_size=14,
        out_chans=256,
        pretrained='SAM-B'#SAM-B SAM-L SAM-H
        ),
    neck = None,
    bbox_head=dict(
        type='SAMHead',
        num_classes=num_classes,
        num_points_perside=10,
        image_size=1024,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        pretrained='SAM-B'#SAM-B SAM-L SAM-H
        ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='SelfNormalize'),
    dict(type='Pad', size=(1024,1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='SelfNormalize'),
            dict(type='Pad', size=(1024,1024)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='tctDataset',
        ann_file='/home/commonfile/tct_ngc_data/annotations/train_image.json',
        img_prefix='/home/commonfile/TCTAnnotated(non-gynecologic)/',
        filter_empty_gt=False,
        pipeline=train_pipeline),
    val=dict(
        type='tctDataset',
        ann_file='/home/commonfile/tct_ngc_data/annotations/val_image.json',
        img_prefix='/home/commonfile/TCTAnnotated(non-gynecologic)/',
        pipeline=test_pipeline),
    test=dict(
        type='tctDataset',
        ann_file='/home/commonfile/tct_ngc_data/annotations/test_image.json',
        img_prefix='/home/commonfile/TCTAnnotated(non-gynecologic)/',
        pipeline=test_pipeline))

optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='step', 
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = 'work_dirs/sam_fcos_tct_ngc_393/byimage/sam_h_backbone'


evaluation = dict(interval=2, metric='bbox')
checkpoint_config = dict(interval=4)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = '/home/ligaojie/mmdetection_2.18.0/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
resume_from = None
load_from = None
workflow = [('train', 1)]