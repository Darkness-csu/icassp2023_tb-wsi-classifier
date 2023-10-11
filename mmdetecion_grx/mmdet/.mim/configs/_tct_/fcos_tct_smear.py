_base_ = [
    './fcos-new_r50_caffe_fpn_gn-head_1x_tct.py'
]
# model settings
model = dict(
    type='FCOS_TCT_SMEAR',
    test_cfg=dict(
        test_feature=True
    )
)
data = dict(
    test=dict(
        img_prefix='../../../commonfile/datasets/TCT_smear/'
    )
)
