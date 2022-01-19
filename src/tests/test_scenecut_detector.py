import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
from tqdm.auto import tqdm

sys.path.append('src')
from scenecut_detector import SceneCutDetector

video_dir = "./input/tensorflow-great-barrier-reef/train_images/video_1/"
video_fns = [os.path.join(video_dir,f'{i}.jpg') for i in range(100000)]
video_fns = [fn for fn in video_fns if os.path.exists(fn)]

scenecut_detector = SceneCutDetector()

for i, fn in tqdm(enumerate(video_fns)):
    
    img = cv2.cvtColor(
        cv2.imread(fn),
        cv2.COLOR_BGR2RGB
    )
    is_cut = scenecut_detector.update_frame(img)
    if is_cut:
        print(f'scene cut detected at {i}, {fn}, prev frame {video_fns[i-1]}')


def derivative(dys):
    return [0] + [dys[i] - dys[i-1] for i in range(1, len(dys))]

plt.plot(
    list(range(len(scenecut_detector.x_movements))),
    np.abs(np.array(derivative(scenecut_detector.x_movements)) + np.abs(np.array(derivative(scenecut_detector.y_movements)))),
    label='dX+dY'
)
plt.plot(list(range(len(scenecut_detector.x_movements))), [70 for _ in range(len(scenecut_detector.x_movements))],
        label='cut_threshold')
plt.legend()

plt.title('Scene cut detection')
plt.ylabel('optical flow abs(dX) + abs(dY)')
plt.xlabel('Frame index')
plt.savefig('./media/scenecut_detection.jpg')
