_base_ = [
    './faster_rcnn_r50_fpn_tct-ngc_cls.py'
]
# model settings
model = dict(
    type='FasterRCNN_TCT_IMAGE',
    test_cfg=dict(
        test_feature=True
    )
)