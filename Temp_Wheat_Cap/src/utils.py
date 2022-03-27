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





def set_plot(size):
    
    """Sets style preferences and text sizes for matplotlib plots."""
    
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.grid']=False
    rcParams['xtick.minor.visible']=True
    rcParams['ytick.minor.visible']=True
    rcParams['xtick.direction']='in'
    rcParams['ytick.direction']='in'

    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size*0.8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size*0.8)    # fontsize of the tick labels
    
    
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


# Transform and preprocess imported df

def preprocess_df(df):
    
    """ Completes preprocessing steps on dataframe to account for column data type correction, bounding box transformations, 
    exploding BoxesString column values, and image name corrections.
    
    Inputs
    - df: name of DataFrame to be preprocessed.
    Returns
    - df: preprocessed version of DataFrame."""
    
    bbox_xmin = []
    bbox_ymin = []
    bbox_width = []
    bbox_height = []
    
    for i, row in df.iterrows():
        row['image_name'] = (row.image_name.split('.')[0]+'.jpg')
        
    for i, row in df.iterrows():
        row['BoxesString'] = list(row['BoxesString'].split(';'))
        
    df = df.explode('BoxesString')
    
    for i, row in df.iterrows():
        row['BoxesString'] = list(row['BoxesString'].split(' '))
        
    df.BoxesString = df.BoxesString.apply(lambda y: list('0 0 0 0'.split(" ")) if len(y)==1 else y)
    
    df_lst = list(df.BoxesString)
    
    for bbox_lst in df_lst:
        bbox_xmin.append(bbox_lst[0])
        bbox_ymin.append(bbox_lst[1])
        bbox_width.append(bbox_lst[2])
        bbox_height.append(bbox_lst[3])
        
    df['bbox_xmin'] = bbox_xmin
    df['bbox_ymin'] = bbox_ymin
    df['bbox_xmax'] = bbox_width
    df['bbox_ymax'] = bbox_height
    
    df.reset_index(inplace=True, drop=True)
    df = df.astype({'bbox_xmin': 'float', 'bbox_ymin': 'float', 'bbox_xmax': 'float', 'bbox_ymax': 'float'})
    
    return df
    


