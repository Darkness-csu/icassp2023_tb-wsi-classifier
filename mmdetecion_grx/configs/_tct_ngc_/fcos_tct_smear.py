_base_ = [
    './fcos-new_r50_caffe_fpn_gn-head_1x_tct-ngc.py'
]
# model settings
model = dict(
    type='FCOS_TCT_SMEAR',
    test_cfg=dict(
        test_feature=True
    )
)
