# Install

Install and activate conda env.

```bash
conda create -n reef --yes python=3.8 jupyter
conda activate reef

conda install -c conda-forge --yes cudnn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Install submodules 

### Yolov5 and YOLOX

```bash
git submodule init && git submodule update

cd yolov5 && pip install -r requirements.txt && cd ..

cd YOLOX && pip install -v -e . && cd ..
```

## Install additional dependencies

```
pip install -r requirements.txt
```

Download [competition data](https://www.kaggle.com/c/tensorflow-great-barrier-reef/data) and place it into `input` folder according to Folder structure below.  

### Folder & file structure
```
root
|_input                  (training data - gitignored)
|  |_tensorflow-great-barrier-reef
|    |_...
|_configs                (configuration files)
|  |_wandb_params.json   (wandb logging config)
|  |_...
|media                   (images etc.)
|  |_...
|_notebooks              (jupyter notebooks)
|  |_...
|_src                    (python files)
|  |_...
|_...
```

