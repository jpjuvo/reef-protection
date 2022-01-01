# Run this python script to create input/train_folds.csv

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold

def count_bin(n):
    """ Bins are selected so that each bin has 5 sequences"""
    if n < 30:
        return 0
    if n < 120:
        return 1
    if n < 500:
        return 2
    return 3

def split_seq_folds(random_state):
    """ Annotation count-bin-stratified split """
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
    
    sequences = []
    for seq in train_df.sequence.unique():
        _df = train_df[train_df['sequence'] == seq]
        n_annotations = np.sum([len(eval(annos)) for annos in _df['annotations'].values])

        sequences.append({
            'sequence' : seq,
            'n_frames' : len(_df),
            'n_annotations' : n_annotations,
            'anno_count_bin' : count_bin(n_annotations),
            'video_id' : _df.video_id.values[0]
        })
    
    seq_df = pd.DataFrame(sequences)
    
    # Split to folds
    seq_df['fold'] = -1
    for fold, (train_index, test_index) in enumerate(skf.split(seq_df.index.values, seq_df.anno_count_bin.values)):
        seq_df.at[test_index,'fold'] = fold
    
    return seq_df

def measure_split_balance(seq_df):
    """ 
    Returns the smallest min-max ratio between fold n_frames or n_anno 
    The closer the value is to 1., the more in balance the folds are
    """
    min_max_frames = (seq_df.groupby(['fold']).sum()['n_frames'].min() / 
                seq_df.groupby(['fold']).sum()['n_frames'].max())

    min_max_anno = (seq_df.groupby(['fold']).sum()['n_annotations'].min() / 
                    seq_df.groupby(['fold']).sum()['n_annotations'].max())
    
    return min(min_max_frames, min_max_anno)
    
train_df = pd.read_csv('./input/tensorflow-great-barrier-reef/train.csv')

# find the most balanced 5-fold split
best_random_seed = 0
highest_min_max = 0
for random_seed in range(10):
    val = measure_split_balance(split_seq_folds(random_seed))
    if val > highest_min_max:
        highest_min_max = val
        best_random_seed = random_seed

# apply the most balanced split
seq_df = split_seq_folds(best_random_seed)

seq_df.to_csv('./input/sequence_stats.csv', index=False)

# apply folds
train_df['fold'] = -1
for seq in train_df.sequence.unique():
    seq_index = train_df[train_df['sequence'] == seq].index
    train_df.at[seq_index, 'fold'] = seq_df[seq_df['sequence'] == seq].fold.values[0]

train_df.to_csv('./input/train_folds.csv', index=False)