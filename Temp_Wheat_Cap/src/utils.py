# import packages
import numpy as np
import os
import pandas as pd 
import pathlib
import PIL
import PIL.Image
import matplotlib.pyplot  as plt
from matplotlib import rcParams
from sklearn.utils import shuffle


def balance_df(df, label='label', class_size=1000):
    
    """Resamples data frame containing class labels so that every class has an equal class size. 
    Classes are sampled with replacement if they exceed the desired class size, and without 
    replacement if they do not.
        
    Inputs
    - df : name of dataframe containing sample data. 
    - label : name of column containing one-hot encoded class labels.
    - class_size : desired size of each class after resampling.
    Returns
    balanced_df - a dataframe with balanced classes."""
    
    balanced_df = pd.DataFrame()
    n_classes = df[label].nunique()
    
    for i in range(n_classes):
        one_class = df[df[label] == i]
        if len(one_class) >= 2000:
            replace=False
        else:
            replace=True
        
        idx = np.random.choice(df[df[label] == i].index, size=class_size, replace=replace)
        temp = df.iloc[idx]
        balanced_df = pd.concat([balanced_df, temp])

    balanced_df = balanced_df.sample(frac=1).reset_index().rename(columns={'index':'old_index'})

    return balanced_df