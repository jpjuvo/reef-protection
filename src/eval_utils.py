import imp
import numpy as np
import cv2
from tqdm.auto import tqdm
from pycocotools import mask as maskUtils

def imagewise_matches(threshold, iou):
    """
    Computes the tp,fp and fn at a given iou_threshold.
    Args:
        threshold (float): iou Threshold.
        iou (np array): IoU matrix. (pred x gts) preds should be in conf descendin order
    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    if iou.ndim < 2:
        return 0,0,0
    
    tp = 0
    fp = 0
    col_indices = list(range(iou.shape[1]))
    used_col_indices = []
    for k, pred_row in enumerate(iou):
        # no more gts left
        if len(col_indices) == 0:
            # add remaining preds to fps and break
            fp += len(iou) - k
            break
        
        # select the max iou for the pred box from free gts
        max_iou = np.max(pred_row[np.array(col_indices)])
        
        # Check the original column index of selection
        pred_row_blocked = pred_row.copy()
        if len(used_col_indices) > 0:
            pred_row_blocked[np.array(used_col_indices)] = -1
        max_iou_col = list(pred_row_blocked).index(max_iou)
        
        # if the iou is above th, it's a match
        if max_iou > threshold:
            tp += 1
            col_indices.remove(max_iou_col)
            used_col_indices.append(max_iou_col)
        else:
            fp += 1
    
    # false negatives are the unmatched gts
    fn = len(col_indices)
    return tp, fp, fn

def f2_score(tp, fp, fn):
    beta = 2
    a = (1+beta**2) * tp
    b = (1+beta**2) * tp + (beta**2)*fn + fp
    return a / b if b != 0 else 1

def precision(tp, fp, fn):
    return tp / (tp + fp + 1e-16)

def recall(tp, fp, fn):
    return tp / (tp + fn + 1e-16)

def f2_at_3_to_8(ious):
    """ 
    Calculates imagewise f2@.3:.8, tps, fps and fns from iou matrix
    
    Returns:
    f2s : f2 scores at each 11 iou threshold
    tps : tp at each 11 iou threshold
    tps : fp at each 11 iou threshold
    fns : fn at each 11 iou threshold
    """
    ths = np.linspace(0.3,0.8,11)
    f2s = []
    tps, fps, fns = [], [], []
    for iou_th in ths:
        tp, fp, fn = imagewise_matches(iou_th, ious)
        f2s.append(f2_score(tp,fp,fn))
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    return np.array(f2s), np.array(tps), np.array(fps), np.array(fns)

def compute_f2(
    val_fns, reef_detector, augment=False,
    count_only_nonempty=False, count_max_samples=None, count_every_nth=1):
    
    imagewise_f2s = []
    all_tps, all_fps, all_fns = [0 for _ in range(11)], [0 for _ in range(11)], [0 for _ in range(11)]
    for i, img_fn in tqdm(enumerate(val_fns[::count_every_nth]), total=len(val_fns)):
        if count_max_samples is not None:
            if i >= count_max_samples:
                break

        img = cv2.cvtColor(
            cv2.imread(img_fn),
            cv2.COLOR_BGR2RGB
        )
        im_h, im_w = img.shape[:2]

        gts = []
        with open(img_fn.replace('/images/','/labels/').replace('.png','.txt').replace('.jpg','.txt')) as f:
            annos = f.readlines()

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
        
        if len(gts) == 0 and count_only_nonempty:
            continue

        pred = reef_detector(
            img, augment=augment)
        
        preds_xywh = [
            [
                p[0], 
                p[1],
                p[2] - p[0],
                p[3] - p[1],
            ] for p in pred
        ]
        
        # box format : x,y,w,h
        ious = maskUtils.iou(preds_xywh, gts, [0]*len(gts))
        if len(ious) == 0:
            f2s = [0 for _ in range(11)]
            tps = [0 for _ in range(11)]
            fps = [0 for _ in range(11)]
            fns = [len(gts) for _ in range(11)]
        else:
            f2s, tps, fps, fns = f2_at_3_to_8(ious)
        for th_i, (tp,fp,fn) in enumerate(zip(tps,fps,fns)):
            all_tps[th_i] += tp
            all_fps[th_i] += fp
            all_fns[th_i] += fn
        
        imagewise_f2s.append(np.mean(f2s))

    imagewise_f2s = np.array(imagewise_f2s)
    competition_f2 = np.mean(
        np.array(
            [f2_score(all_tps[th_i], all_fps[th_i], all_fns[th_i]) for th_i in range(11)]
        ))
    return competition_f2, imagewise_f2s