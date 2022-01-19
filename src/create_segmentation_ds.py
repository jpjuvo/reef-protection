
import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import sys

sys.path.append('src')
from boxtractor import Boxtractor
from reef_detector import ReefDetector

def exists_or_make_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


CROP_DS_DIR = './input/crop_ds/'
CROP_IM_DIR = os.path.join(CROP_DS_DIR, 'images')
CROP_LABEL_DIR = os.path.join(CROP_DS_DIR, 'labels')
exists_or_make_dir(CROP_DS_DIR)
exists_or_make_dir(CROP_IM_DIR)
exists_or_make_dir(CROP_LABEL_DIR)

def get_iou(pred_box, gt_box):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Modified from
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    pred_box : enumerable of [x1,y1,x2,y2]
    gt_box : enumerable of [x1,y1,w,h]

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {
        'x1' : pred_box[0],
        'y1' : pred_box[1],
        'x2' : pred_box[2],
        'y2' : pred_box[3],
        }
    bb2 = {
        'x1' : gt_box[0],
        'y1' : gt_box[1],
        'x2' : gt_box[0] + gt_box[2],
        'y2' : gt_box[1] + gt_box[3],
        }

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main(fold:int, yolo_pth:str, threshold:float=0.2, im_size:int=1280):

    dir_path = './input/yolo_ds/'
    with open(dir_path + f'{fold}_val_images.txt') as file:
        lines = file.readlines()
        val_fns = [os.path.join(dir_path, fn.replace('\n','')) for fn in lines]

    reef_detector = ReefDetector(
        weights=yolo_pth,
        im_size=im_size,
        conf_thres=threshold,
        device='cuda'
    )

    image_dicts = []
    boxtractor = Boxtractor()

    for i in tqdm(range(len(val_fns))):
        img_fn = val_fns[i]
        img = cv2.cvtColor(
            cv2.imread(img_fn),
            cv2.COLOR_BGR2RGB
        )

        with open(img_fn.replace('/images/','/labels/').replace('.png','.txt').replace('.jpg','.txt')) as f:
            annos = f.readlines()

        im_h, im_w = img.shape[:2]
        gts = []
        for anno in annos:
            lbl, xc, yc, w, h = anno.split(' ')
            xc = float(xc) * im_w
            yc = float(yc) * im_h
            w = float(w) * im_w
            h = float(h) * im_h
            x = (xc - w / 2)
            y = (yc - h / 2)
            gts.append([
                x,y,w,h
            ])
            
        basename = os.path.basename(img_fn).replace('.jpg', '')
        
        pred = reef_detector(img, augment=True)
        pred_crops, anno_masks, areas = boxtractor.boxtract(img, pred, annos=gts)
        
        for j, (crop, mask, area) in enumerate(zip(pred_crops, anno_masks, areas)):
            cv2.imwrite(os.path.join(CROP_IM_DIR, basename + f'-{j}.jpg'), cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(CROP_LABEL_DIR, basename + f'-{j}.jpg'), mask)
            
            iou = 0.0
            # measure iou
            for gt in gts:
                iou = max(iou, get_iou(pred[j], gt))

            image_dicts.append(
                {
                    'id' : basename + f'-{j}',
                    'width' : pred[j][2] - pred[j][0],
                    'height' : pred[j][3] - pred[j][1],
                    'center_x' : int(pred[j][0] + (pred[j][2] - pred[j][0]) // 2),
                    'center_y' : int(pred[j][1] + (pred[j][3] - pred[j][1]) // 2),
                    'area' : area,
                    'fold' : fold,
                    'iou' : iou
                }
            )
        
    crop_df = pd.DataFrame(image_dicts)
    crop_df.to_csv(os.path.join(CROP_DS_DIR, f'train-fold-{fold}.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[])
    
    parser.add_argument('--fold', type=int)
    parser.add_argument('--yolo_pth', type=str)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--im_size', type=int, default=1280)

    params = parser.parse_args()
    fold = params.fold
    yolo_pth = params.yolo_pth
    threshold = params.threshold
    im_size = params.im_size
    
    main(fold, yolo_pth, threshold, im_size)