_base_ = [
    './cas_rram_gram.py'
]

model = dict(
    type='FS_FasterRCNN_SMEAR',
    smear_cfg=dict(
        test_feature=True,
        in_index = 0,
        in_channel = 256,
        mid_channel = 6
    )
)