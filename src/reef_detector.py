import os
import numpy as np
import cv2
import torch
import sys
import pandas as pd
import torch

sys.path.append('.')
sys.path.append('..')
from yolov5.models.yolo import Model as YoloModel
from yolov5.utils.torch_utils import scale_img

def load_hub_model(path, conf=0.2, iou=0.50, max_det=100, hub_path='./yolov5/'):
    model = torch.hub.load(hub_path,
                           'custom',
                           path=path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = max_det  # maximum number of detections per image
    return model

def voc2coco(bboxes, image_height=720, image_width=1280):
    bboxes  = voc2yolo(bboxes, image_height, image_width)
    bboxes  = yolo2coco(bboxes, image_height, image_width)
    return bboxes

def voc2yolo(bboxes, image_height=720, image_width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    return bboxes

def yolo2coco(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    """
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    return bboxes

def _forward_augment_reef(self, x):
    """ Custom augmentation routine for yolo that's adjusted for reef competition """
    img_size = x.shape[-2:]  # height, width
    s = [1, 1.1, 1.2]  # scales
    f = [None, 3, None]  # flips (2-ud, 3-lr)
    y = []  # outputs
    for si, fi in zip(s, f):
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        yi = self._forward_once(xi)[0]  # forward
        # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)
    y = self._clip_augmented(y)  # clip augmented tails
    return torch.cat(y, 1), None  # augmented inference, train

class ReefDetector:
    
    def __init__(self, 
                 weights, 
                 im_size=1280, 
                 device=None, 
                 conf_thres=0.2,
                 iou_thres=0.5,
                 max_det=100,
                 hub_path='./yolov5/'):
        
        self.device = device
        self.im_size = im_size
        self.model = load_hub_model(
            path=weights, 
            conf=conf_thres, 
            iou=iou_thres, 
            max_det=max_det,
            hub_path=hub_path
        )
        self.model.to(device)
        
        # custom augmentation routine
        self.model.model.model._forward_augment = _forward_augment_reef.__get__(
            self.model.model.model,
            YoloModel
        )
        
    def __call__(self, np_im, 
                 augment=False
                ):
        """ 
        Returns model predictions from rgb numpy image
        
        Returns
        sorted_pred_list : list of x1,y1,x2,y2,conf in conf descending order
        """
        
        height,width = np_im.shape[:2]
        results = self.model(np_im, size=self.im_size, augment=augment)
        
        preds   = results.pandas().xyxy[0]
        bboxes  = preds[['xmin','ymin','xmax','ymax']].values
        if len(bboxes) > 0:
            bboxes  = voc2coco(bboxes,height,width).astype(int)
            confs   = preds.confidence.values
        else:
            bboxes = []
            confs = []
            
        # Sort to descending confidence order
        # make sure confs are unique before sorting
        confs = list(confs)
        confs = np.array([c if confs.count(c) == 1 else c + (i*0.0001) for i,c in enumerate(confs)])
        if len(bboxes) > 0:
            confs, bboxes = zip(*sorted(zip(confs, bboxes), reverse=True))
        
        # combine to boxes and scores
        sorted_preds = [
            [box[0],box[1],box[0]+box[2],box[1]+box[3], conf] for box,conf in zip(bboxes, confs)]
        
        return sorted_preds