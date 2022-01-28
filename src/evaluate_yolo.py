import os
import sys
import numpy as np
import cv2
import sys
import pandas as pd
from tqdm.auto import tqdm
import argparse

sys.path.append('.')
from src.reef_detector import ReefDetector
from src.eval_utils import compute_f2

dir_path = './input/yolo_ds/'

def main(fold:int, yolo_pth:str, threshold:float=0.2, im_size:int=1280):
    
    with open(dir_path + f'{fold}_val_images.txt') as file:
        lines = file.readlines()
        val_fns = [os.path.join(dir_path, fn.replace('\n','')) for fn in lines]
    
    reef_detector = ReefDetector(
        weights=yolo_pth,
        im_size=im_size,
        conf_thres=threshold,
        device='cuda',
        hub_path='./yolov5/'
    )

    competition_f2, _ = compute_f2(
        val_fns, reef_detector, augment=False,
        count_max_samples=None, count_every_nth=1)

    print(f'F2 score: {competition_f2:.4f}')
    
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