_base_ = [
    './faster_rcnn_r50_fpn_tct_cls.py'
]
# model settings
model = dict(
    type='FasterRCNN_TCT_SMEAR',
    test_cfg=dict(
        test_feature=True
    ),
    cls_head=dict(
        in_channels=256,
        mid_channels=10,
        in_index=0,
        loss_weight=0.3,
        pos_list=[1, 2, 3, 4, 5]
    )
)
data = dict(
    test=dict(
        img_prefix='../../../commonfile/datasets/TCT_smear/'
    )
)
