_base_ = [
    './fcos-new_r50_caffe_fpn_gn-head_1x_tct.py'
]
# model settings
model = dict(
    type='FCOS_TCT_SMEAR',
    test_cfg=dict(
        test_feature=True
    ),
    cls_head=dict(
        in_channels=256,
        in_index=0,
        loss_weight=0.3,
        pos_list=[1, 2, 3, 4, 5]
    ),
    # cls_head=None,
)
data = dict(
    test=dict(
        img_prefix='../../../commonfile/datasets/TCT_smear/'
    )
)
