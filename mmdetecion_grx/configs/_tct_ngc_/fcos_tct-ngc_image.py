_base_ = [
    './fcos_tct-ngc_cls.py'
]
# model settings
model = dict(
    type='FCOS_TCT_IMAGE',
    test_cfg=dict(
        cls_result=True
    )
)
