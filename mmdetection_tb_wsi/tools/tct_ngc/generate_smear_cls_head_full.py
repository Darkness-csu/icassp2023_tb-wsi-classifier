import argparse
import os
import numpy as np
import random
from functools import partial
from multiprocessing import Pool

import torch
from tqdm import tqdm

#ROOT = '../data/TCT/smear/'
ROOT = '../../../commonfile/processed/TCT_NGC_DETR/new_smear_cls_head_v4/'
# (path, type)
INPUT_LIST = [
    (ROOT + 'Annotated20211207XIMEATCT-FFK--FYB_WSI', 'neg'),
    #(ROOT + 'Annotated20211207XIMEATCT-FFK--FYB-12358', 'neg'),
    #(ROOT + 'TCTFFK-BJXK-20210311/Negative', 'neg'),
    #(ROOT + 'TCTFFK-BJXK-20210311/Positive', 'pos'),
    (ROOT + 'Annotated20220209XIMEATCT-FFK---10294', 'pos'),
    (ROOT + 'Annotated20220212XIMEATCT-FFK--FYB', 'neg'),
    (ROOT + 'Annotated20220915XIMEATCT-FFK-NILM', 'neg'),
    (ROOT + 'Annotated20220915XIMEATCT-FFK-POS', 'pos'),
    (ROOT + 'TCT_NGC-BJXK-20210910', 'mix')
    #(ROOT + 'Unannotated20220629HBSYYTCT-FFK-238', 'mix')
]
folder_suffix = ['/DigitalSlice', '/OriginalImage']
neg_keywords = ['NILM', 'JPXB']

output_path = '../../../commonfile/processed/TCT_NGC_DETR/full_v4/full'
classes = ['neg', 'pos']
file_suffix = '.pt'

def Split_Seq(seq):
    splitNum = seq[0]
    seq = seq[1:]#两个部分都不包含中间值，因此切片去除seq[0]
    theBig = [x for x in seq if x >= splitNum]
    theSmall = [x for x in seq if x < splitNum]
    return splitNum,theBig,theSmall
#找出中间值
def topKNum(seq,k):
    splitNum, theBig, theSmall = Split_Seq(seq)
    theBigLen = len(theBig)
    
    if  k == theBigLen:
        return splitNum#出口,返回这个中间值,
    
    if k > theBigLen:
        return topKNum(theSmall,k-theBigLen-1)
    # 大值的列表中大于K个数的情况
    return topKNum(theBig,k)
#由中间值找出TopK个值，<list>
def getTopK(seq,k):
    if k == len(seq):
        return seq
    num = 0
    result = []
    mid = topKNum(seq, k)
    for i in seq :
        if i >= mid and num < k:
            result.append(i)
            num += 1
    return result

def getTopKIndex(seq,k):
    if k == len(seq):
        return list(range(len(seq)))
    num = 0
    result_index = []
    mid = topKNum(seq, k)
    for index,value in enumerate(seq) :
        if value >= mid and num < k:
            result_index.append(index)
            num += 1
    return result_index

def generate_smear_pt(path):#这里输入的是单张smear的文件夹路径
    output = None
    num = 200
    all_images = []
    all_cls_results = []
    for file in os.listdir(path):
        if file.endswith(file_suffix):
            try:
                image = torch.load(os.path.join(path, file), map_location='cpu')
                all_images.append(image[:-1])
                all_cls_results.append(image[-1])
                #image = image.unsqueeze(0)

            except Exception as e:
                print(f'Error in {path}: {e}')
                continue

    if len(all_images) < num: #不够数目的得重复采样
        total_num = len(all_images)
        index_list = list(range(total_num))
        group_num = int(num/total_num)
        sample_num = num%total_num
        samples = random.sample(index_list, sample_num)
        result_index = index_list*group_num + samples
    else:
        result_index = getTopKIndex(all_cls_results,num)
    
    for index in result_index:
        image = all_images[index]
        image = torch.Tensor(image).unsqueeze(0)
        if output is None:
            output = image
        else:
            output = torch.cat((output,image),0)

    return output
    #num += 1
    # output = output / num
    # output = torch.stack(output, dim=0)
    # output = torch.mean(output, dim=0)
    # print(num, output.shape, path)
  


def process(x):
    key, value = x
    path = value['path']
    ann_type = value['ann_type']
    file_name = f'{key}.pt'

    output_file = os.path.join(output_path, ann_type, file_name)

    if not os.path.exists(output_file):
        output = generate_smear_pt(path)
        torch.save(output, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=0)
    args = parser.parse_args()

    smear_list = dict()

    for path, ann_type in INPUT_LIST:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(file_suffix):
                    file_path = os.path.join(root, file)

                    smear_folder = os.path.dirname(file_path)      #smear_folder表示的是特征图文件的目录路径
                    smear_name = str(smear_folder)                 #smear_name是去掉了smear_folder中间的'/DigitalSlice', '/OriginalImage'
                    for suffix in folder_suffix:
                        smear_name = smear_name.replace(suffix, '')
                    smear_name = os.path.basename(smear_name)

                    if smear_name in smear_list.keys():
                        exist_folder = smear_list[smear_name]['path']
                        print(f'WARNING: duplicate name {smear_name}: '
                              f'\n{os.path.relpath(smear_folder, ROOT)}  <->  {os.path.relpath(exist_folder, ROOT)}\n')
                        break

                    if ann_type != 'mix':
                        smear_ann_type = ann_type
                    else:
                        smear_ann_type = 'pos'
                        for s in neg_keywords:
                            if s in smear_name:
                                smear_ann_type = 'neg'

                    smear_list[smear_name] = dict(
                        path=smear_folder,
                        ann_type=smear_ann_type
                    )
                    break

    for c in classes:
        os.makedirs(os.path.join(output_path, c), exist_ok=True)

    if args.j <= 0:
        for key, value in tqdm(smear_list.items(), ascii=True):
            path = value['path']
            ann_type = value['ann_type']
            file_name = f'{key}.pt'

            output_file = os.path.join(output_path, ann_type, file_name)

            if not os.path.exists(output_file):
                output = generate_smear_pt(path)
                if output != None:
                    torch.save(output, output_file)
            # else:
            #     print(f'skip {output_file}')
    else:
        with Pool(args.j) as p:
            list(tqdm(
                p.imap(partial(process), smear_list.items()), total=len(smear_list.items()), ascii=True
            ))
            p.close()
