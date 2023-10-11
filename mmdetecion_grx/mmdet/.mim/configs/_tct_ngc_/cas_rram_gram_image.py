_base_ = [
    './cas_rram_gram_cls.py'
]
# model settings
model = dict(
    type='FasterRCNN_TCT_IMAGE',
)