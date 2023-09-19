import argparse
import datetime
import imghdr
import json
import os
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

ROOT = '../../../commonfile/TCTAnnotated(non-gynecologic)/'
PROCESSED_ROOT = '../../../commonfile/processed/TCT_NGC_DETR_961/new_smear_cls_head_v4/'

KSJ_SMEAR_ROOT = '/home/commonfile/TCTAnnotated(non-gynecologic)/Unannotated_KSJ/'
XIMEA_SMEAR_ROOT = '/home/commonfile/TCTAnnotated(non-gynecologic)/Unannotated_XIMEA/'
INPUTS_LIST = [
    KSJ_SMEAR_ROOT + 'Unannotated-KSJ-TCTNGC-NILM',
    KSJ_SMEAR_ROOT + 'Unannotated-KSJ-TCTNGC-POS',
    XIMEA_SMEAR_ROOT + 'Unannotated-XIMEA-TCTNGC-NILM',
    XIMEA_SMEAR_ROOT + 'Unannotated-XIMEA-TCTNGC-POS'
]

classes_name_to_id = {
    #'mesothelial_cell': 1,
    #'mesothelial_cell': 7,
    # 'blood_cell': 0,
    # 'tissue_cell': 0,
    'adenocarcinoma': 2,            #腺癌
    'squamous_cell_carcinoma': 3,   #鳞癌
    'small_cell_carcinoma': 4,      #小细胞癌
    # 'lymphoma_rare': 0,
    'mesothelioma_common': 5,       #间皮瘤
    'diseased_cell': 6,             #病变细胞
    'NILM': 7                       #阴性细胞
    #supplement
    #'JPXB': 1,
    #'JPXB': 7, 
    #'XA': 2
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=120)
    parser.add_argument('--output', type=str, default='/home/commonfile/tct_ngc_data/annotations/detr_test_smear.json')
    parser.add_argument('-j', type=int, default=0)
    parser.add_argument('--no-check', action='store_true')
    args = parser.parse_args()
    return args


def process(x):
    file_name, label_id = x
    if not imghdr.what(os.path.join(ROOT, file_name)):
        print(f'skip broken file: {file_name}')
        return None
    else:
        return x


def get_file_list(path_list, label_id, j=0, no_check=False):
    file_list = []

    skip_num = 0

    for path in path_list:
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                
                file_name = os.path.relpath(os.path.join(root, file), ROOT)

                if PROCESSED_ROOT is not None and \
                        os.path.exists(os.path.join(PROCESSED_ROOT, file_name.replace('.jpg', '.pt'))):
                    skip_num += 1
                    continue

                file_list.append([file_name, label_id])
    print(f'Skip {skip_num} processed files')

    if no_check:
        return file_list

    final_file_list = []
    if j <= 0:
        for file_name, label_id in file_list:
            if not imghdr.what(os.path.join(ROOT, file_name)):
                print(f'skip broken file: {file_name}')
                continue

            final_file_list.append([file_name, label_id])
    else:
        with Pool(j) as p:
            final_file_list = list(tqdm(
                p.imap(partial(process), file_list), total=len(file_list), ascii=True
            ))
            p.close()

    final_file_list = [x for x in final_file_list if x is not None]
    return final_file_list


def init_label_dict():
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=None,
            contributor=None,
            date_created=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='detection',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    return data


def generate_json_cls(files, classes_name_to_id):
    json_file = init_label_dict()

    categories = []
    for class_name, id_ in classes_name_to_id.items():
        categories.append(dict(
            supercategory=None,
            id=id_,
            name=class_name,
        ))

    json_file['categories'] = categories

    image_id = 1
    ann_id = 1
    for x in files:
        image_file, category = x[0], int(x[1])
        if 'KSJ' in image_file:
            json_file['images'].append(dict(
                license=0,
                url=None,
                file_name=image_file,
                height=1024,
                width=1792,
                date_captured=None,
                id=image_id,
            ))
        else:
            json_file['images'].append(dict(
                license=0,
                url=None,
                file_name=image_file,
                height=2816,
                width=4096,
                date_captured=None,
                id=image_id,
            ))
        json_file['annotations'].append(dict(
            id=ann_id,
            image_id=image_id,
            category_id=1,
            segmentation=[[]],
            area=100 * 100,
            bbox=[0, 0, 100, 100],
            iscrowd=0,
        ))

        image_id += 1
        ann_id += 1

    return json_file


if __name__ == '__main__':
    args = parse_args()

    ROOT = os.path.abspath(ROOT)

    print(f'Seed {args.seed}')
    print(f'Root {ROOT}')
    print(f'Inputs: {INPUTS_LIST}\n')

    file_list = get_file_list(INPUTS_LIST, label_id=1, j=args.j, no_check=args.no_check)
    print(f'Files: {len(file_list)}')

    test_json = generate_json_cls(file_list, classes_name_to_id)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(test_json, f)

    print(f"Save json files to {args.output}")
