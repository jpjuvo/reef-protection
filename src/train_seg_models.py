import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import cv2
import os
import sys
import torch
import pandas as pd
import seaborn as sns
import fastai
from fastai.vision.all import *
from fastai.vision.gan import *
from PIL import ImageDraw, ImageFont
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import linear_model
import gc

DIR_DS = './input/crop_ds/'
DIR_CROP = os.path.join(DIR_DS, 'images')
DIR_LABELS = os.path.join(DIR_DS, 'labels')

# Use end part of fold 0 as test set
test_df = pd.read_csv(f'./input/crop_ds/train-fold-0.csv')
test_df['frame_index'] = [int(s.split('-')[1]) for s in test_df['id'].values]
test_df = test_df[test_df['frame_index'] > 9186]
val_df = None

def is_val(fn):
    is_val = os.path.basename(fn).replace('.jpg','') in list(val_df['id'].values)
    is_test = os.path.basename(fn).replace('.jpg','') in list(test_df['id'].values)
    return is_val or is_test

def get_dls(bs:int=32, size:int=128, fold=0):
    global val_df
    val_df = pd.read_csv(f'./input/crop_ds/train-fold-{fold}.csv')
    
    # leave test part out
    if fold == 0:
        val_df['frame_index'] = [int(s.split('-')[1]) for s in val_df['id'].values]
        val_df = val_df[val_df['frame_index'] <= 9186]

    dblock = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        get_items=get_image_files,
        get_y = lambda x: os.path.join(DIR_LABELS, x.name),
        #splitter=RandomSplitter(),
        splitter=FuncSplitter(is_val),
        item_tfms=Resize(size),
        batch_tfms=[*aug_transforms(
            min_zoom=0.8,
            max_zoom=2.,
            max_lighting=0.3,
            flip_vert=True,
            
        ),
                    Normalize.from_stats(*imagenet_stats)])
    dls = dblock.dataloaders(DIR_CROP, bs=bs, path='.')
    dls.c = 3 # For 1 channel image
    return dls

def create_gen_learner(dls):
    return unet_learner(
        dls, 
        arch=resnet18,#resnet34, 
        lr=0.001,
        wd=1e-3,
        loss_func = MSELossFlat(),              
        blur=True, 
        norm_type=NormType.Weight, 
        self_attention=True)

def plot_preds(learn_gen, index = 0, fold=0):
    f, axs = plt.subplots(1,4, figsize=(10,4))
    img_fn = os.path.join(DIR_CROP, val_df['id'].values[index] + '.jpg')
    mask_fn = os.path.join(DIR_LABELS, val_df['id'].values[index] + '.jpg')
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_fn)
    pred = learn_gen.predict(img)

    pred_tta_0 = learn_gen.predict(img)[0].detach().numpy()[0]
    pred_tta_1 = learn_gen.predict(img[:,::-1,:])[0].detach().numpy()[0,:,::-1]
    pred_tta = np.mean(np.stack([pred_tta_0, pred_tta_1], 0), 0)

    axs[0].imshow(img)
    axs[0].set_title('Prediction box')
    axs[1].imshow(mask)
    axs[1].set_title('GT starfish')
    axs[2].imshow(pred[0].detach().numpy()[0])
    axs[2].set_title('Pred')
    axs[3].imshow(pred_tta)
    axs[3].set_title('Pred TTA avg.')

    plt.savefig(f'./media/segmentation_sample_fold-{fold}.jpg')
    plt.close('all')

def extract_values(pred):
    # center crop
    margin=24
    crop = pred[margin:-margin,margin:-margin]
    mean_val = np.mean(crop)
    max_val = np.max(crop)
    center = int(crop[crop.shape[0]//2, crop.shape[1]//2])
    return mean_val, max_val, center

def split_dfs_to_data(dfs, x_columns=[
    'width', 'height', 'box_size', 'pred_mean', 'pred_max', 'pred_center'
], target_col='iou', fold=0):
    train_df = pd.concat([df for i, df in enumerate(dfs) if i != fold], axis=0, ignore_index=True)
    val_df = dfs[fold]

    train_xs = train_df[x_columns].values
    train_ys = train_df[target_col].values
    
    val_xs = val_df[x_columns].values
    val_ys = val_df[target_col].values
    
    return train_xs, train_ys, val_xs, val_ys

def main():

    dfs = []
    for fold in range(5):
        print(f'Fold {fold}')
        dls = get_dls(bs=32, fold=fold)

        mixup = MixUp()
        learn_gen = create_gen_learner(dls)

        learn_gen.fit_one_cycle(2, pct_start=0.8, wd=1e-3, cbs=mixup)

        learn_gen.unfreeze()
        learn_gen.fit_one_cycle(4, slice(1e-6,1e-3), wd=1e-3, cbs=mixup)
        torch.save(learn_gen.model.state_dict(), f'./output/starfish_seg_resnet18_unet_fold-{fold}.pth')

        plot_preds(learn_gen, index = 0, fold=fold)

        mean_vals = []
        max_vals = []
        center_vals = []
        for i in tqdm(range(len(val_df))):
            img_fn = os.path.join(DIR_CROP, val_df['id'].values[i] + '.jpg')
            img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
                
            pred_tta_0 = learn_gen.predict(img)[0].detach().numpy()[0]
            pred_tta_2 = learn_gen.predict(img[:,::-1,:])[0].detach().numpy()[0,:,::-1]
            pred_tta = np.mean(np.stack([pred_tta_0, pred_tta_2], 0), 0)

            mean_val, max_val, center_val = extract_values(pred_tta)
            
            mean_vals.append(mean_val)
            max_vals.append(max_val)
            center_vals.append(center_val)
        
        learn_gen = None
        dls = None
        gc.collect()
        torch.cuda.empty_cache()

        val_df['pred_mean'] = mean_vals
        val_df['pred_max'] = max_vals
        val_df['pred_center'] = center_vals
        val_df['box_size'] = val_df['width'].values * val_df['width'].values
        val_df['gt'] = val_df['iou'].values > 0.1
        dfs.append(val_df.copy())

    oof_preds = []
    oof_gts = []

    print('Fitting linreg')
    x_columns = ['box_size', 'pred_mean']
    coefs = []
    for fold in range(5):
        train_xs, train_ys, val_xs, val_ys = split_dfs_to_data(dfs, x_columns=x_columns, fold=fold)
        reg = linear_model.LinearRegression()
        #reg = linear_model.BayesianRidge()
        reg.fit(train_xs, train_ys)
        val_pred = reg.predict(val_xs)
        oof_gts += list(val_ys)
        oof_preds += list(val_pred)
        print(reg.coef_)
        coefs.append(reg.coef_)
    
    coefs = np.mean(np.array(coefs), 0)
    print(f'Avg coefs {coefs}')

    # sort in pred ascending order
    oof_preds, oof_gts = zip(*sorted(zip(oof_preds, oof_gts)))
    max_gts = []
    mean_gts = []
    perc_95_gts = []
    data_percents = []
    for i, (p, gt) in enumerate(zip(oof_preds, oof_gts)):
        max_gts.append(max(oof_gts[:i+1]))
        mean_gts.append(np.mean(oof_gts[:i+1]))
        perc_95_gts.append(np.percentile(oof_gts[:i+1], 95))
        data_percents.append(i / len(oof_preds))

    f, axs = plt.subplots(2,1,figsize=(10,20))
    df_plot = pd.DataFrame({'Pred': np.array(oof_preds), 'iou': np.array(oof_gts)})
    sns.regplot(
        data=df_plot, 
        x = 'Pred', 
        y = 'iou', 
        fit_reg = False, 
        x_jitter = 0.01, 
        y_jitter = 0.01,
        ax=axs[0])

    df_plot = pd.DataFrame({
        'Pred_th': np.array(oof_preds), 
        'max_iou': np.array(max_gts),
        'mean_iou': np.array(mean_gts),
        'perc95_iou': np.array(perc_95_gts),
        'boxes_percentage' : np.array(data_percents)
    })

    for p,m,perc in zip(oof_preds, max_gts, data_percents):
        if m > 0.1:
            ax_title = f'{perc*100:.1f}% under th:{p:.2f} where GT-iou is under 0.1'
            print(ax_title)
            break
    
    sns.lineplot(data=df_plot, x = 'Pred_th', y = 'max_iou', label='max_iou', ax=axs[1])
    sns.lineplot(data=df_plot, x = 'Pred_th', y = 'mean_iou', label='mean_iou', ax=axs[1])
    sns.lineplot(data=df_plot, x = 'Pred_th', y = 'perc95_iou', label='perc95_iou', ax=axs[1])
    sns.lineplot(data=df_plot, x = 'Pred_th', y = 'boxes_percentage', label='boxes_percentage', ax=axs[1])
    axs[1].set_title(ax_title)

    plt.savefig(f'./media/bbox_segmentation_feature_linreg.jpg')
    plt.close('all')


if __name__ == '__main__':
    main()