import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import cv2
import os

def plot_yolo_sample(
        img_fn, 
        colors=['orange', 'green','red','yellow'],
        labels=None, ax=None):
    im = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    with open(img_fn.replace('/images/','/labels/').replace('.png','.txt').replace('.jpg','.txt')) as f:
        annos = f.readlines()
    
    annos = [x.strip() for x in annos]
    
    if ax is None:
        f, ax = plt.subplots(1,1)
    ax.imshow(im)
    
    im_h, im_w = im.shape[:2]
    
    for anno in annos:
        lbl, x, y, w, h = anno.split(' ')
        color = colors[int(lbl) % len(colors)]
        label_txt = labels[int(lbl)] if labels is not None else ""
        
        x = int(float(x) * im_w)
        y = int(float(y) * im_h)
        w = int(float(w) * im_w)
        h = int(float(h) * im_h)
        
        rect = patches.Rectangle(
            (x-w//2, y-h//2),
            w,h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        
        ax.text(x-w//2 + 7, y-h//2 + 26, label_txt, bbox=dict(facecolor=color, alpha=0.5), fontsize=12)
        ax.add_patch(rect)
        ax.set_title(os.path.basename(img_fn))
    
def plot_yolo_pred(img_fn, preds, ax):
    
    im = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_h, im_w = im.shape[:2]

    with open(img_fn.replace('/images/','/labels/').replace('.png','.txt').replace('.jpg','.txt')) as f:
        annos = f.readlines()

    if ax is None:
        f, ax = plt.subplots(1,1)
    ax.imshow(im)

    for anno in annos:
        lbl, x, y, w, h = anno.split(' ')
        color = 'green'
        
        x = int(float(x) * im_w)
        y = int(float(y) * im_h)
        w = int(float(w) * im_w)
        h = int(float(h) * im_h)
        
        rect = patches.Rectangle(
            (x-w//2, y-h//2),
            w,h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )        
        ax.add_patch(rect)
    
    for pred in preds:
        x1,y1,x2,y2,conf = pred[:5]
        label_txt = f'{conf:.3f}'
        color = 'orange'
        x = int(round(x1))
        y = int(round(y1))
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        
        rect = patches.Rectangle(
            (x, y),
            w,h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        
        ax.text(x + 7, y - 16, label_txt, bbox=dict(facecolor=color, alpha=0.5), fontsize=12)
        ax.add_patch(rect)