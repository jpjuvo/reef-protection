# This creates yolo format yaml configs to input/yolo_ds
# symlinks training images and creates yolo annotation files 

import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy

def exists_or_make_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

SAVE_ROOT = './input/yolox_ds/'
FOLD_DIRS = [os.path.join(SAVE_ROOT, f'fold-{fold}') for fold in range(5)]
FOLD_TRAIN_DIRS = [os.path.join(d, 'train2017') for d in FOLD_DIRS]
FOLD_VAL_DIRS = [os.path.join(d, 'val2017') for d in FOLD_DIRS]
FOLD_ANNO_DIRS = [os.path.join(d, 'annotations') for d in FOLD_DIRS]


ROOT_VIDEOS = './input/tensorflow-great-barrier-reef/train_images/'

exists_or_make_dir(SAVE_ROOT)
for i,fold_dir in enumerate(FOLD_DIRS):
    exists_or_make_dir(fold_dir)
    exists_or_make_dir(FOLD_TRAIN_DIRS[i])
    exists_or_make_dir(FOLD_VAL_DIRS[i])
    exists_or_make_dir(FOLD_ANNO_DIRS[i])

df = pd.read_csv('./input/train_folds.csv')

# 40 is better than 160 (better recall)
# 10 is better than 40
# 10 is better than 1
BG_KEEP_EVERY_N = 5
BG_COUNTER = 0

annotion_id = 0

annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
annotations_json['info'].append({
        "year": "2022",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "-",
        "date_created": "2022-02-02T12:00:00+00:00"
    })
annotations_json["licenses"].append({
            "id": int(1),
            "url": "",
            "name": "Unknown"
        })
annotations_json["categories"].append({
    "id": 0, "name": "starfish", "supercategory": "none"
    })

fold_train_annotations = [copy.deepcopy(annotations_json) for _ in range(5)]
fold_val_annotations = [copy.deepcopy(annotations_json) for _ in range(5)]

for idx in tqdm(range(len(df))):
    video_id = df.iloc[idx]['video_id']
    frame_id = df.iloc[idx]['video_frame']
    fold = int(df.iloc[idx]['fold'])
    annos = eval(df.iloc[idx]['annotations'])

    # keep only every nth bg sample
    if len(annos) == 0:
        BG_COUNTER += 1
        if BG_COUNTER % BG_KEEP_EVERY_N != 0:
            continue
        BG_COUNTER = 0

    img_fn = os.path.join(
        ROOT_VIDEOS, 
        f'video_{video_id}', 
        f'{frame_id}.jpg'
    )
    im_h, im_w = cv2.imread(img_fn).shape[:2]

    image_id = df.iloc[idx]['image_id']
    image_json = {
            "id": frame_id,
            "license": int(1),
            "file_name": f'{image_id}.jpg',
            "height": int(im_h),
            "width": int(im_w),
            "date_captured": "2022-02-02T12:00:00+00:00"
        }

    # symlink images to new places
    for i, (train_dir, val_dir) in enumerate(zip(FOLD_TRAIN_DIRS, FOLD_VAL_DIRS)):
        new_fn = os.path.join(train_dir, image_id + ".jpg") if i != fold else os.path.join(val_dir, image_id + ".jpg")
        if not os.path.exists(new_fn):
            os.symlink(os.path.abspath(img_fn), new_fn)

        if i != fold:
            fold_train_annotations[i]['images'].append(image_json)
        else:
            fold_val_annotations[i]['images'].append(image_json)


    for anno in annos:
        x,y = int(anno['x']), int(anno['y'])
        w,h = int(anno['width']), int(anno['height'])
        w = int(min(im_w - x - 1, w))
        h = int(min(im_h - y - 1, h))

        annotation_json = {
                "id": int(annotion_id),
                "image_id": frame_id,
                "category_id": int(0),
                "bbox": [x, y, w, h],
                "area": int(w * h),
                "segmentation": [],
                "iscrowd": int(0)
            }
        annotion_id += 1

        for i in range(5):
            if i != fold:
                fold_train_annotations[i]['annotations'].append(annotation_json)
            else:
                fold_val_annotations[i]['annotations'].append(annotation_json)

# save training and val file paths
for fold in range(5):
    
    with open(os.path.join(FOLD_ANNO_DIRS[fold], f'train.json'), 'w') as f:
        out_json = fold_train_annotations[fold]
        f.write(json.dumps(out_json, cls=NpEncoder))

    with open(os.path.join(FOLD_ANNO_DIRS[fold], f'valid.json'), 'w') as f:
        out_json = fold_val_annotations[fold]
        f.write(json.dumps(out_json, cls=NpEncoder))
