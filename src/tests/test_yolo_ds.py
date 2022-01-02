import matplotlib.pyplot as plt
import sys

sys.path.append('src')
from yolo_utils import plot_yolo_sample

plot_yolo_sample('./input/yolo_ds/images/0-50.jpg')
plt.show()