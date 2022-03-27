import pandas as pd
import numpy as np
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import cv2

# setup an example augmentation pipeline

example_transforms = A.Compose([
    A.RandomSizedBBoxSafeCrop(512, 512, erosion_rate=0.0, interpolation=1, p=1.0),
    A.HorizontalFlip(p=0.75),
    A.VerticalFlip(p=0.75),
    A.OneOf([A.HueSaturationValue(),
             A.Sharpen(),
             A.RandomToneCurve(always_apply=True, scale=0.3),
             A.ColorJitter(),
             A.RandomBrightnessContrast(),
             A.RGBShift()], p=1.0),
    A.OneOf([A.RandomSunFlare(p=0.6, src_radius=50),
             A.RandomRain(p=0.6, blur_value=2, rain_type='heavy'),
             A.RandomFog(p=0.6,fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.8),
             A.RandomShadow(p=0.6, num_shadows_lower=2),
             A.MotionBlur(p=0.6)], p= 1.0),
    A.CLAHE(p=1.0)], p=1.0, 
    bbox_params = A.BboxParams(format='coco', min_visibility=0.1, label_fields=['category_id']))


# define a function to apply a random set of transformations to a random image from the dataset

def apply_transforms(transforms, df, n_transforms=3, figsize = (10,10)):
    idx = np.random.randint(len(df), size=1)[0]
    
    image_id = df.iloc[idx].image_name
    bboxes = []
    for _, row in df[df.image_name == image_id].iterrows():
        bboxes.append([row.bbox_xmin, row.bbox_ymin, row.bbox_xmax-row.bbox_xmin, row.bbox_ymax-row.bbox_ymin])
        
    image = Image.open(data_dir / image_id)
    
    fig, axs = plt.subplots(1, n_transforms+1, figsize=figsize)
    
    # plot the original image
    axs[0].imshow(image)
    axs[0].set_title('original')
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
        axs[0].add_patch(rect)
    
    # apply transforms n_transforms times
    for i in range(n_transforms):
        params = {'image': np.asarray(image),
                  'bboxes': bboxes,
                  'category_id': [1 for j in range(len(bboxes))]}
        augmented_boxes = transforms(**params)
        bboxes_aug = augmented_boxes['bboxes']
        image_aug = augmented_boxes['image']

        # plot the augmented image and augmented bounding boxes
        axs[i+1].imshow(image_aug)
        axs[i+1].set_title('augmented_' + str(i+1))
        for bbox in bboxes_aug:
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
            axs[i+1].add_patch(rect)
    plt.show()