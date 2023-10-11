import argparse
import os
import shutil

import numpy as np

label_to_id = {
    'neg': 0,
    'pos': 1
}


def split(files, train_ratio, seed):
    np.random.seed(seed)

    file_list = np.array(files, dtype=object)

    index = np.array(range(len(file_list)))
    np.random.shuffle(index)

    train_index = index[:int(len(index) * train_ratio)]
    test_index = index[len(train_index):]

    train_list = file_list[sorted(train_index)].tolist()
    test_list = file_list[sorted(test_index)].tolist()

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

    train_list = file_list[sorted(train_index)].tolist()
    val_list = file_list[sorted(val_index)].tolist()
    test_list = file_list[sorted(test_index)].tolist()

    return train_list, val_list, test_list

def generate(files, output_root, split):
    lines = []

    shutil.rmtree(os.path.join(output_root, split), ignore_errors=True)

    for file, path, label in files:
        lines.append('{} {}\n'.format(os.path.join(label, file), label_to_id[label]))

        output_path = os.path.join(output_root, split, label, file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(path, output_path)

    ann_file = os.path.join(output_root, f'{split}.txt')
    with open(ann_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'Save to {ann_file}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input', type=str, default='../../../commonfile/processed/TCT_NGC/fcos_smear_full/full')
    parser.add_argument('--output', type=str, default='../../../commonfile/processed/TCT_NGC/fcos_smear_full/')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    file_list = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            label = os.path.basename(root)
            path = os.path.join(root, file)
            file_list.append((file, path, label))

    #train_list, test_list = split(file_list, train_ratio=args.train_ratio, seed=args.seed)
    train_list, val_list, test_list = new_split(file_list, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    print(f'Train ratio: {args.train_ratio}')
    print(f'Val ratio: {args.val_ratio}')
    print(f'Seed: {args.seed}')
    #print(f'Train: {len(train_list)}  Test: {len(test_list)}')
    print(f'Train: {len(train_list)}  Val:{len(val_list)} Test: {len(test_list)}')

    generate(train_list, args.output, split='train')
    generate(val_list, args.output, split='val')
    generate(test_list, args.output, split='test')
    generate(file_list, args.output, split='all')
