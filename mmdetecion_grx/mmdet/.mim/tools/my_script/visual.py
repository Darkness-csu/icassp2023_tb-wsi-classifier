import os

import cvtools

if __name__ == '__main__':
    output = './visual'
    os.makedirs(os.path.join(output, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output, 'test'), exist_ok=True)

    tool = cvtools.label_analysis.COCOAnalysis(
        img_prefix='../data/TCTAnnotated(non-gynecologic)/TCT_NGC-BJXK-20210910',
        ann_file='../data/TCT/annotations/TCT_NGC-BJXK-20210910/train.json'
    )
    tool.vis_instances(os.path.join(output, 'train'), vis='bbox')

    tool = cvtools.label_analysis.COCOAnalysis(
        img_prefix='../data/TCTAnnotated(non-gynecologic)/TCT_NGC-BJXK-20210910',
        ann_file='../data/TCT/annotations/TCT_NGC-BJXK-20210910/test.json'
    )
    tool.vis_instances(os.path.join(output, 'test'), vis='bbox')