_base_ = [
    './faster_rcnn_r50_fpn_tct_cls.py'
]
# model settings
model = dict(
    type='FasterRCNN_TCT_SMEAR',
    test_cfg=dict(
        test_feature=True
    )
)
data = dict(
    test=dict(
        img_prefix='../../../commonfile/datasets/TCT_smear/'
    )
)
