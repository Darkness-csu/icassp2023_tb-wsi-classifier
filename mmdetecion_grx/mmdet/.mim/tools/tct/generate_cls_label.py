import argparse
import datetime
import json
import os

import numpy as np

ROOT = '/home/commonfile/TCTAnnotated(non-gynecologic)/'
POS_LIST = [
    ROOT + 'POS'
]
NEG_LIST = [
    ROOT + 'NILM'
]
classes_name_to_id = {
    "normal": 1,
    "ascus": 2,
    "asch": 3,
    "lsil": 4,
    "hsil_scc_omn": 5,
    "agc_adenocarcinoma_em": 6,
    "vaginalis": 7,
    "monilia": 8,
    "dysbacteriosis_herpes_act": 9,
    "ec": 10
}
# classes_name_to_id = {
#     'negative': 1,
#     'positive': 2,
# }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=120)
    parser.add_argument('--output', type=str, default='../data/TCT/annotations/TCT_NGC')
    args = parser.parse_args()
    return args


def get_file_list(path_list, label_id):
    file_list = []

    for path in path_list:
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                file_name = os.path.relpath(os.path.join(root, file), ROOT)
                file_list.append([file_name, label_id])
    return file_list


def split(files, train_ratio, seed):
    np.random.seed(seed)

    file_list = np.array(files, dtype=object)

    index = np.array(range(len(file_list)))
    np.random.shuffle(index)

    train_index = index[:int(len(index) * train_ratio)]
    test_index = index[len(train_index):]

    train_list = file_list[train_index].tolist()
    test_list = file_list[test_index].tolist()

    return train_list, test_list


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
        # print(category, ' ', image_file)
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
            category_id=category,
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

    print(f'Seed {args.seed}')
    print(f'Root {ROOT}')
    print(f'Positive: {POS_LIST}')
    print(f'Negative: {NEG_LIST}')

    pos_list = get_file_list(POS_LIST, label_id=2)
    neg_list = get_file_list(NEG_LIST, label_id=1)
    train_pos, test_pos = split(pos_list, train_ratio=0.8, seed=args.seed)
    train_neg, test_neg = split(neg_list, train_ratio=0.8, seed=args.seed)

    train_list = []
    test_list = []
    train_list.extend(train_pos)
    train_list.extend(train_neg)
    test_list.extend(test_pos)
    test_list.extend(test_neg)

    train_json = generate_json_cls(train_list, classes_name_to_id)
    test_json = generate_json_cls(test_list, classes_name_to_id)

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'train_cls.json'), 'w') as f:
        json.dump(train_json, f, indent=2)

    with open(os.path.join(args.output, 'test_cls.json'), 'w') as f:
        json.dump(test_json, f, indent=2)

    print(f'Positive: {len(train_pos)}  {len(test_pos)}')
    print(f'Negative: {len(train_neg)}  {len(test_neg)}')

    print(f"Save json files to {args.output}")
