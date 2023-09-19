import argparse
import os
import numpy as np
from functools import partial
from multiprocessing import Pool

import torch
from tqdm import tqdm

ROOT = '../../../commonfile/processed/TCT_NGC_DETR_961/smear_cls_head_query/'

INPUT_LIST = [
    (ROOT + 'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM', 'neg'),
    (ROOT + 'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS', 'pos'),
    (ROOT + 'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM', 'neg'),
    (ROOT + 'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS', 'pos')
   
   
]

output_path = '../../../commonfile/processed/TCT_NGC_DETR_961/smear_cls_head_query_full/full'
classes = ['neg', 'pos']
file_suffix = '.pt'

def Split_Seq(seq):
    #print(len(seq))
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
    all_querys = []
    all_cls_scores = []
    
    for file in os.listdir(path):
        if file.endswith(file_suffix):
            try:
                querys = torch.load(os.path.join(path, file), map_location='cpu')
                for i in range(len(querys)):
                    all_querys.append(querys[i,:-2])
                    all_cls_scores.append(querys[i,-2])

            except Exception as e:
                print(f'Error in {path}: {e}')
                continue

    if len(all_cls_scores) < num:
        return None
    
    #print(len(all_cls_scores))
    result_index = getTopKIndex(all_cls_scores,num)
    
    for index in result_index:
        query = all_querys[index]
        query = torch.Tensor(query).unsqueeze(0)
        if output is None:
            output = query
        else:
            output = torch.cat((output,query),0)

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
                    
                    smear_name = os.path.basename(smear_name)

                    if smear_name in smear_list.keys():
                        exist_folder = smear_list[smear_name]['path']
                        print(f'WARNING: duplicate name {smear_name}: '
                              f'\n{os.path.relpath(smear_folder, ROOT)}  <->  {os.path.relpath(exist_folder, ROOT)}\n')
                        break

                    
                    smear_ann_type = ann_type
                    

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
