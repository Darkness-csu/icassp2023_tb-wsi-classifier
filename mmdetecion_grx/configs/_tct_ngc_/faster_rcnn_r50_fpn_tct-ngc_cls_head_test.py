_base_ = [
    './faster_rcnn_r50_fpn_tct-ngc_cls.py'
]
# model settings
model = dict(
    type='FasterRCNN_TCT_CLS',
    test_cfg=dict(
        test_cls=True
    )
)