import datetime
import json
import os

import cv2
import numpy as np
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

ROOT = '/home/commonfile/TCTAnnotated(non-gynecologic)/'
INPUTS = [
    (ROOT + 'TCT_NGC-BJXK-20210910', 'txt'),
    (ROOT + 'Annotated20211207XIMEATCT-FFK--FYB-12358', 'xml'),
    (ROOT + 'Annotated20220209XIMEATCT-FFK---10294', 'xml')
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=124)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--output', type=str, help='output path of annotations',
                        default='../data/TCT_NGC/annotations')#../data/TCT/annotations/TCT_NGC 
    args = parser.parse_args()
    return args


def get_txt_image_list(path):
    file_list = []
    for file in os.listdir(path):
        if not file.endswith('.txt'):
            continue
        file_list.append(os.path.join(path, file))

    image_dict = dict()
    for file in file_list:
        ann_type, ann_num, ann_list = load_txt(file, path_prefix='OriginalImage/')

        image_root = file.replace('.txt', '')

        # for ann in tqdm(ann_list):
        for ann in ann_list:
            category, image_file, x, y, w, h = ann

            if w < 0 or h < 0:
                continue

            image_path = os.path.join(image_root, image_file)
            file_name = os.path.relpath(image_path, ROOT)

            if not os.path.exists(image_path):
                print(f'WARNING: Not Found {image_path}')
                continue

            if image_path not in image_dict.keys():
                image_dict[image_path] = [(category, file_name, x, y, w, h)]
            else:
                image_dict[image_path].append((category, file_name, x, y, w, h))

        # print(f'TXT {file} {len(ann_list)}')

    return list(image_dict.values())


def get_xml_image_list(path):
    image_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.xml'):
                continue

            image_path = os.path.join(root, file.replace('.xml', '.jpg'))
            file_name = os.path.relpath(image_path, ROOT)

            if not os.path.exists(image_path):
                print('WARNING: Not Found {}'.format(image_path))
                continue

            ann_list = load_xml(os.path.join(root, file), file_name)
            image_list.append(ann_list)

            # print(f'XML {file} {len(ann_list)}')
    return image_list


def load_txt(file, has_head=True, path_prefix='OriginalImage/'):
    with open(file, 'r', encoding='utf-8') as f:
        if has_head:
            ann_type = int(f.readline())  # 玻片类型

        ann_num = int(f.readline().rstrip())  # 标注框总数量

        ann_list = []
        for i, row in enumerate(f):
            row_data = row.split()
            category = int(row_data[0])
            image = path_prefix + "{}_{}.jpg".format(row_data[1], row_data[2])
            x = int(row_data[3])
            y = int(row_data[4])
            w = int(row_data[5])
            h = int(row_data[6])

            ann_list.append([category, image, x, y, w, h])

    return ann_type, ann_num, ann_list


def load_xml(file, img_file_name):
    res = []

    tree = ET.parse(file)

    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))
    if height != 2816 and width != 4096:
        print(f"WARNING: {height} x {width} is not matching this code, {file}")

    for obj in tree.iter("object"):
        category = obj.findtext('name')
        xmin = int(obj.findtext("bndbox/xmin"))
        ymin = int(obj.findtext("bndbox/ymin"))
        xmax = int(obj.findtext("bndbox/xmax"))
        ymax = int(obj.findtext("bndbox/ymax"))

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        if w <= 0 or h <= 0:
            continue

        res.append((category, img_file_name, x, y, w, h))

    return res


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


def new_split(files, train_ratio, val_ratio, seed):
    np.random.seed(seed)

    file_list = np.array(files, dtype=object)

    index = np.array(range(len(file_list)))
    np.random.shuffle(index)

    train_index = index[:int(len(index) * train_ratio)]
    rest_index = index[len(train_index):]
    val_index = rest_index[:int(len(index) * val_ratio)]
    test_index = rest_index[len(val_index):]

    train_list = file_list[train_index].tolist()
    val_list = file_list[val_index].tolist()
    test_list = file_list[test_index].tolist()

    return train_list, val_list, test_list



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


def generate_json(image_files, classes_name_to_id, id_reflect):
    json_file = init_label_dict()

    categories = []
    for class_name, id_ in classes_name_to_id.items():
        categories.append(dict(
            supercategory=None,
            id=id_,
            name=class_name,
        ))

    json_file['categories'] = categories

    image_file_dict = dict()

    image_id = 0
    ann_id = 0
    for ann_list in image_files:

        # for ann in tqdm(ann_list):
        for ann in ann_list:
            category, file_name, x, y, w, h = ann

            if file_name not in image_file_dict.keys():
                image_file_dict[file_name] = image_id
                json_file['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=file_name,
                    # height=int(img_data.shape[0]),
                    # width=int(img_data.shape[1]),
                    height=2816,
                    width=4096,
                    date_captured=None,
                    id=image_id,
                ))
                image_id += 1

            json_file['annotations'].append(dict(
                id=ann_id,
                image_id=image_file_dict[file_name],
                category_id=id_reflect[category],
                segmentation=[[]],
                area=w * h,
                bbox=[x, y, w, h],
                iscrowd=0,
            ))
            ann_id += 1

    return json_file


def analysis(path):
    with open(path) as f:
        ann = json.load(f)

    categories = ann['categories']

    num = dict()

    for c in categories:
        num[c['id']] = 0

    for ann in ann['annotations']:
        label = ann['category_id']
        num[label] += 1

    print(path)
    for k, v in num.items():
        print(k, v)


if __name__ == '__main__':
    args = parse_args()

    print(f'Seed {args.seed}')
    print(f'Root {ROOT}')
    print(f'Inputs: {INPUTS}')

    classes_name_to_id = {
        #'mesothelial_cell': 1,
        #'mesothelial_cell': 7,
        # 'blood_cell': 0,
        # 'tissue_cell': 0,
        'adenocarcinoma': 2,
        'squamous_cell_carcinoma': 3,
        'small_cell_carcinoma': 4,
        # 'lymphoma_rare': 0,
        'mesothelioma_common': 5,
        'diseased_cell': 6,
        'NILM': 7,

        #supplement
        #'JPXB': 1,
        #'JPXB': 7, 
        'XA': 2
    }
    id_reflect = {
        #30: 1,
        30: 7,
        # 31: 0,
        # 32: 0,
        33: 2,
        34: 3,
        35: 4,
        # 36: 0,
        37: 5,
        38: 6,

        #supplement
        #'JPXB': 1,
        'JPXB': 7, 
        'XA': 2,
        'NILM': 7,
        'zsjp':7,
        'ZSJP':7,
        'XXBA':4,
        'JPL':5

    }


    '''
    image_list = []

    for path, ann_type in INPUTS:
        if ann_type == 'txt':
            images = get_txt_image_list(path)
            image_list.extend(images)
        elif ann_type == 'xml':
            images = get_xml_image_list(path)
            image_list.extend(images)
        else:
            raise NotImplementedError

    train_list, test_list = split(image_list, args.train_ratio, args.seed)
    
    '''
    train_list = []
    val_list = []
    test_list = []

    for path, ann_type in INPUTS:
        if ann_type == 'txt':
            images = get_txt_image_list(path)
            t_train_list, t_val_list, t_test_list = new_split(images, args.train_ratio, args.val_ratio, args.seed)
            train_list.extend(t_train_list)
            val_list.extend(t_val_list)
            test_list.extend(t_test_list)
        elif ann_type == 'xml':
            images = get_xml_image_list(path)
            t_train_list, t_val_list, t_test_list = new_split(images, args.train_ratio, args.val_ratio, args.seed)
            train_list.extend(t_train_list)
            val_list.extend(t_val_list)
            test_list.extend(t_test_list)
        else:
            raise NotImplementedError



    train_json = generate_json(train_list, classes_name_to_id, id_reflect)
    val_json = generate_json(val_list, classes_name_to_id, id_reflect)
    test_json = generate_json(test_list, classes_name_to_id, id_reflect)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'new_train.json'), 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(args.output, 'new_val.json'), 'w') as f:
        json.dump(val_json, f, indent=2)
    with open(os.path.join(args.output, 'new_test.json'), 'w') as f:
        json.dump(test_json, f, indent=2)

    print(f"Save json files to {args.output}")
    print(len(train_list)+len(val_list)+len(test_list))
    print(len(train_list), len(val_list), len(test_list))

    analysis(os.path.join(args.output, 'new_train.json'))
    analysis(os.path.join(args.output, 'new_val.json'))
    analysis(os.path.join(args.output, 'new_test.json'))
