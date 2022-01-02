# This creates yolo format yaml configs to input/yolo_ds
# symlinks training images and creates yolo annotation files 

import os
import cv2
import numpy as np
import pandas as pd
import random
import yaml
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from yolo_utils import plot_yolo_sample

def exists_or_make_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

SAVE_ROOT = './input/yolo_ds/'
IMG_DIR = os.path.join(SAVE_ROOT, 'images')
ANNO_DIR = os.path.join(SAVE_ROOT, 'labels')

ROOT_VIDEOS = './input/tensorflow-great-barrier-reef/train_images/'

exists_or_make_dir(SAVE_ROOT)
exists_or_make_dir(IMG_DIR)
exists_or_make_dir(ANNO_DIR)
df = pd.read_csv('./input/train_folds.csv')

fold_fns = [[] for _ in range(5)]

for idx in tqdm(range(len(df))):
    video_id = df.iloc[idx]['video_id']
    frame_id = df.iloc[idx]['video_frame']
    fold = int(df.iloc[idx]['fold'])
    img_fn = os.path.join(
        ROOT_VIDEOS, 
        f'video_{video_id}', 
        f'{frame_id}.jpg'
    )
    im_h, im_w = cv2.imread(img_fn).shape[:2]

    image_id = df.iloc[idx]['image_id']
    new_fn = os.path.join(IMG_DIR, image_id + ".jpg")

    fold_fns[fold].append(new_fn)

    # create a symlink for image
    os.symlink(os.path.abspath(img_fn), new_fn)

    # create yolo annotations
    label_fn = os.path.join(ANNO_DIR, image_id + '.txt')
    annos = eval(df.iloc[idx]['annotations'])
    
    with open(label_fn, 'w') as f:
        for anno in annos:
            x,y = int(anno['x']), int(anno['y'])
            w,h = int(anno['width']), int(anno['height'])

            # scale to 0-1 coordinates
            x,y,w,h = (
                np.array([x,y,w,h]).astype(np.float) /
                np.array([im_w, im_h, im_w, im_h]).astype(np.float)
                )
            xc = x + w / 2
            yc = y + h / 2

            line = f'0 {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}'
            f.write(line + '\n')

# save training and val file paths
for fold in range(5):
    
    train_fns = []
    val_fns = []
    for i in range(5):
        if i != fold:
            train_fns += fold_fns[i].copy()
        else:
            val_fns += fold_fns[i].copy()
    
    # shuffle training file order
    random.shuffle(train_fns)

    with open(os.path.join(SAVE_ROOT,f'{fold}_train_images.txt'), 'w') as f:
        for fn in train_fns:
            f.write(f'{ os.path.join("images", os.path.basename(fn)) }\n')
    
    with open(os.path.join(SAVE_ROOT,f'{fold}_val_images.txt'), 'w') as f:
        for fn in val_fns:
            f.write(f'{ os.path.join("images", os.path.basename(fn)) }\n')

    yaml_fn = os.path.join(SAVE_ROOT, f'fold_{fold}.yaml')
    if os.path.exists(yaml_fn):
        os.remove(yaml_fn)
        
    dict_file = {
        'train' : os.path.join('..', SAVE_ROOT, f'{fold}_train_images.txt'),
        'val' : os.path.join('..', SAVE_ROOT, f'{fold}_val_images.txt'),
        'test' : os.path.join('..', SAVE_ROOT, f'{fold}_val_images.txt'),
        'nc' : 1,
        'names' : [
            'starfish'
        ]
    }
    
    with open(yaml_fn, 'w') as file:
        yaml.dump(
            dict_file, 
            file, 
            default_flow_style=False)

# Check that conversion worked
plot_yolo_sample(fold_fns[0][10])
plt.show()