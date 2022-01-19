import os
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2

matplotlib.use('tkAgg')

sys.path.append('src')
from boxtractor import Boxtractor

image_fn = "./input/tensorflow-great-barrier-reef/train_images/video_0/1919.jpg"
img = cv2.cvtColor(cv2.imread(image_fn), cv2.COLOR_BGR2RGB)
pred = [
    [1127, 246, 1207, 310, 0.9442859126091003],
    [716, 459, 781, 514, 0.9441859126091003],
    [1197, 417, 1274, 480, 0.563419759273529],
    [1081, 151, 1170, 216, 0.2553739845752716]
]

gts = [
    [1125.056, 248.004, 66.944, 58.968],
    [727.04, 466.02, 50.944, 43.992000000000004]
]

boxtractor = Boxtractor()
pred_crops, anno_masks, areas = boxtractor.boxtract(img, pred, annos=gts)

f, axs = plt.subplots(3, len(pred), figsize=(10, 8))
for i in range(len(pred)):
    axs[0,i].imshow(pred_crops[i])
    axs[1,i].imshow(anno_masks[i])
    
    axs[2,i].imshow(pred_crops[i])
    axs[2,i].imshow(anno_masks[i], alpha=0.5)

    axs[0,i].set_title(f'Pred box {i}')
    axs[1,i].set_title(f'Starfish area:{areas[i]:.2f}')
    axs[2,i].set_title(f'Overlay')

plt.savefig('./media/boxtractor.jpg')