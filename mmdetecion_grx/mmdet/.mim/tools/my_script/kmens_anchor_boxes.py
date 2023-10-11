# origin: https://github.com/lars76/kmeans-anchor-boxes

import json
import os

import numpy as np
import argparse


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        print(box[0], box[1], clusters[:, 0], clusters[:, 1])
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def load_dataset(path_list):
    dataset = []

    for path in path_list:
        with open(path) as f:
            label = json.load(f)
        images = label['images']
        annotations = label['annotations']

        for ann in annotations:
            x, y, w, h = ann['bbox']
            image = images[ann['image_id']]
            width = image['width']
            height = image['height']
            dataset.append([w / width, h / height])

    return np.array(dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path of annotation file')
    parser.add_argument('-k', type=int, default=5, help='num of clusters')
    parser.add_argument('--width', type=int, default=1312, help='input width during training')
    parser.add_argument('--height', type=int, default=800, help='input height during training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ann_path_list = []

    if args.input.endswith('.json'):
        ann_path_list.append(args.input)
    else:
        for file in os.listdir(args.input):
            if file.endswith('.json'):
                ann_path_list.append(os.path.join(args.input, file))

    print('Input:\n {}'.format(ann_path_list))

    data = load_dataset(ann_path_list)
    out = kmeans(data, k=args.k)

    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

    bbox = out * np.array([args.width, args.height])
    print("Boxes for input {} x {}:\n {}".format(args.width, args.height, bbox))
