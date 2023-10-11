# ref: fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py

_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_tct-ngc.py'

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.6)))

optimizer_config = dict(_delete_=True, grad_clip=None)

lr_config = dict(warmup='linear')

work_dir = 'new_work_dirs/fcos_tct-ngc/new_protocol'