import numpy as np
import cv2

class Boxtractor:
    """ 
    Extracts crops from image where the predictions or annotations are,
    and resizes to square fixed size
    """
    
    def __init__(self, box_size=128, padding=8):
        self.box_size = box_size
        self.padding = padding
        
    def boxtract(self, img, preds, annos=None):
        """
        Parameters:
            img (ndarray) : rgb image
            preds (list) : list of pred arrays [x1,y1,x2,y2,score]
            annos (list) : optional list of gt arrays [x1,y1,w,h]
            
        Returns:
            pred_crops (list) : prediction crop images
            anno_masks (list) : only returned if annos are not None, 
                                segmentation masks of crops where 0=BG,255=FG
            areas (list) : only returned if annos are not None, 
                           areas that FGs cover in masks [0-1]
        """
        gt_mask = None
        if annos is not None:
            gt_mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
            for anno in annos:
                x1,y1,w,h = anno
                x1 = int(round(x1))
                y1 = int(round(y1))
                xc = int(round(x1+w/2))
                yc = int(round(y1+h/2))
                w = int(round(w))
                h = int(round(h))
                cv2.ellipse(gt_mask,(xc,yc),(w//2,h//2),0,0,360,255,-1)
                #gt_mask[y1:y1+h, x1:x1+w] = 255
            gt_mask = cv2.blur(gt_mask, (15,15))
        
        pred_crops, anno_masks, areas = [],[],[]
        img_h, img_w = img.shape[:2]
        for i, pred in enumerate(preds):
            x1,y1,x2,y2 = pred[:4]
            x1 = np.clip(int(round(x1)) - self.padding, 0, img_w - 1)
            x2 = np.clip(int(round(x2)) + self.padding, 0, img_w - 1)
            y1 = np.clip(int(round(y1)) - self.padding, 0, img_h - 1)
            y2 = np.clip(int(round(y2)) + self.padding, 0, img_h - 1)
            
            if x1 == x2 or y1 == y2:
                continue
            
            crop = img[y1:y2, x1:x2, :]
            
            if gt_mask is not None:
                crop_mask = gt_mask[y1:y2, x1:x2]
                areas.append(
                    (np.sum(np.where(crop_mask > 128,1,0)) / (crop_mask.shape[0]*crop_mask.shape[1]))
                )
                anno_masks.append(
                    cv2.resize(crop_mask, (self.box_size, self.box_size) ,cv2.INTER_NEAREST))
            
            pred_crops.append(
                cv2.resize(crop, (self.box_size, self.box_size) ,cv2.INTER_LINEAR))
        
        if annos is not None:
            return pred_crops, anno_masks, areas
        return pred_crops