from string import Template
import argparse
import requests
import os

PIPELINE_CONFIG_PATH='../configs/yolox_last_config.py'
template_fn = '../configs/yolox_s_config.py'

def main(fold=0, epochs=10, img=1280):

    with open(template_fn, 'r') as f:
        config_file_template = f.read()

    pipeline = Template(config_file_template).substitute(max_epoch=epochs)
    pipeline = Template(pipeline).substitute(data_dir=f'../input/yolox_ds/fold-{fold}')
    pipeline = Template(pipeline).substitute(input_size=(img, img))
    pipeline = Template(pipeline).substitute(test_size=(img, img))

    with open(PIPELINE_CONFIG_PATH, 'w') as f:
        f.write(pipeline)

    voc_cls = '''VOC_CLASSES = ( "starfish" ,)'''
    with open('../YOLOX/yolox/data/datasets/voc_classes.py', 'w') as f:
        f.write(voc_cls)
    coco_cls = '''COCO_CLASSES = ("starfish",)'''
    with open('../YOLOX/yolox/data/datasets/coco_classes.py', 'w') as f:
        f.write(coco_cls)
    
    MODEL_DIR = '../input/yolox_weights'
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    MODEL_PATH = os.path.join(MODEL_DIR, 'yolox_s.pth')
    if not os.path.isfile(MODEL_PATH):
        url = 'https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth'
        r = requests.get(url, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img', type=int, default=1280)

    params = parser.parse_args()
    fold = params.fold
    epochs = params.epochs
    img = params.img
    
    main(fold=fold, epochs=epochs, img=img)