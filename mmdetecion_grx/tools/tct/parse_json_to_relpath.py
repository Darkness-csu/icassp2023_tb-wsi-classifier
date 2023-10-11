import json


def process(path):
    with open(path) as f:
        ann = json.load(f)

    for i in range(len(ann['images'])):
        ann['images'][i]['file_name'] = ann['images'][i]['file_name'].replace('/root/commonfile/TCTAnnotatedData/', '')

    with open(path, 'w') as f:
        json.dump(ann, f, indent=2)


if __name__ == '__main__':
    process('D:/file/server/root/userfolder/data/TCT/annotations/train.json')
    process('D:/file/server/root/userfolder/data/TCT/annotations/val.json')
    process('D:/file/server/root/userfolder/data/TCT/annotations/test.json')
