import argparse
import os
from functools import partial
from multiprocessing import Pool

import torch
from tqdm import tqdm

ROOT = '../../../commonfile/processed/TCT/smear/'
# (path, type)
INPUT_LIST = [
    (ROOT + 'NILM', 'neg'),
    (ROOT + 'POS', 'pos')
]
folder_suffix = ['/DigitalSlice', '/OriginalImage', '/TCT']
neg_keywords = ['NILM']

output_path = '../../../commonfile/processed/TCT/smear_full/full'
classes = ['neg', 'pos']
file_suffix = '.pt'


def generate_smear_pt(path):
    output = None
    num = 0

    for file in os.listdir(path):
        if file.endswith(file_suffix):
            try:
                image = torch.load(os.path.join(path, file), map_location='cpu')
            except Exception as e:
                print(f'Error in {path}: {e}')
                continue

            if output is None:
                output = image
            else:
                output += image

            num += 1
    output = output / num
    # output = torch.stack(output, dim=0)
    # output = torch.mean(output, dim=0)
    # print(num, output.shape, path)
    return output


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

                    smear_folder = os.path.dirname(file_path)
                    smear_name = str(smear_folder)
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
                torch.save(output, output_file)
            else:
                print(f'skip {output_file}')
    else:
        with Pool(args.j) as p:
            list(tqdm(
                p.imap(partial(process), smear_list.items()), total=len(smear_list.items()), ascii=True
            ))
            p.close()
