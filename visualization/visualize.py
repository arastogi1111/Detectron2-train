from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import os
import copy
import matplotlib.patches as mpatches
from tqdm import tqdm

from utils.bb_viz import getBBs, draw_bboxes, draw_bboxes_only
from utils.classes_ids import get_class_dict

threshold = 0.6


def visualize_submission_preds(csv_submission, test_dir = None, samples = 20, threshold = 0.6):
    # Enter your kaggle submission csv
    sub = pd.read_csv(csv_submission)

    save_dir_main = str(csv_submission)
    save_dir_main = save_dir_main[:save_dir_main.rfind(".csv")]
    save_dir_main = Path(save_dir_main)
    
   # Common directory with preprocessed original sized JPGs 
    if test_dir is None:
        test_dir = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/datasets/original_jpg/vinbigdata/test"

    # Where to save annotated images
    save_dir = save_dir_main / f"test_output_imgs_conf_th={threshold}"
    save_dir_noconf = save_dir_main / f"test_output_imgs_th={threshold}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_noconf, exist_ok=True)

    class_dict = get_class_dict()


    cmap = "Spectral"
    pal = [tuple([int(x) for x in np.array(c)*(255,255,255)]) for c in sns.color_palette(cmap, 14)]


    print("Drawing Annotations on Images ...")

    row_i = 0
    number_of_finding_samples = 0

    while number_of_finding_samples < samples:
        row_i+=1
        
        image_id = sub.image_id.iloc[row_i]
        img = cv2.imread(f"{test_dir}/{image_id}.jpg")
        img2 = copy.deepcopy(img)
        bbs = getBBs(sub.iloc[row_i] , th=threshold)
        
        if len(bbs)==0:
            continue
        else:
            number_of_finding_samples+=1

        path_set  = set()
        handles = []
        
        for bb in bbs:
            labels, scores, xmin, ymin, xmax, ymax = tuple(bb)
            
            img2 = draw_bboxes_only(img2, (xmin,ymin), (xmax,ymax), rgb=pal[labels], opacity=0.08, line_thickness=4)
            path_set.add(labels)
            
            img = draw_bboxes(img,(xmin,ymin),(xmax,ymax), rgb=pal[labels], label=class_dict[labels], conf = scores, opacity=0.08, line_thickness=4)
            
        for labels in path_set:    
            handles.append(mpatches.Patch(color=tuple(np.array(pal[labels])/255), label=class_dict[labels]))  
        
        fig,ax = plt.subplots(figsize =(22,30))
        plt.imshow(img2)
        plt.tight_layout()
        ax.legend(handles=handles, prop={'size': 20})
        ax.axis('off')
        img2 = img2.astype('uint8')
        fig.savefig(save_dir_noconf / f'{image_id}_annotated.jpeg')
        plt.close(fig)

        plt.figure(figsize =(30,20))
        plt.imshow(img)
        img = img.astype('uint8')
        plt.imsave(save_dir / f'{image_id}_annotated_conf.jpeg', img)
        plt.close()




    # if you want corresponding orig imgs too

    # from shutil import copy2

    # orig_img_dir = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/datasets/original_jpg/vinbigdata/test"
    # copy_img_dir = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/detectron2/test_orig_imgs"

    # for row_i in range(0,20):
    #     image_id = sub.image_id.iloc[row_i]
    #     copy2(f"{orig_img_dir}/{image_id}.jpg", f"{copy_img_dir}/{image_id}.jpg")


def visualize_QA_preds(csv_qa, qa_dir = None, threshold = 0.6):
    # Enter your kaggle submission csv
    sub = pd.read_csv(csv_qa)

    save_dir_main = str(csv_qa)
    save_dir_main = save_dir_main[:save_dir_main.rfind(".csv")]
    save_dir_main = Path(save_dir_main)
    
   # Common directory with preprocessed original sized JPGs 
    if qa_dir is None:
        qa_dir = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/datasets/original_jpg/vinbigdata/train"

    # Where to save annotated images
    save_dir = save_dir_main / f"QA_output_imgs_conf_th={threshold}"
    save_dir_noconf = save_dir_main / f"QA_output_imgs_th={threshold}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_noconf, exist_ok=True)

    class_dict = get_class_dict(True)


    cmap = "Spectral"
    pal = [tuple([int(x) for x in np.array(c)*(255,255,255)]) for c in sns.color_palette(cmap, 15)]


    print("Drawing Annotations on Images ...")

    
    number_of_finding_samples = 0

    for row_i in tqdm(range(len(sub))):
        
        image_id = sub.image_id.iloc[row_i]
        img = cv2.imread(f"{qa_dir}/{image_id}.jpg")
        img2 = copy.deepcopy(img)
        bbs = getBBs(sub.iloc[row_i] , th=threshold)
        
        

        path_set  = set()
        handles = []
        
        for bb in bbs:
            labels, scores, xmin, ymin, xmax, ymax = tuple(bb)
            
            img2 = draw_bboxes_only(img2, (xmin,ymin), (xmax,ymax), rgb=pal[labels], opacity=0.08, line_thickness=4)
            path_set.add(labels)
            
            img = draw_bboxes(img,(xmin,ymin),(xmax,ymax), rgb=pal[labels], label=class_dict[labels], conf = scores, opacity=0.08, line_thickness=4)
            
        for labels in path_set:    
            handles.append(mpatches.Patch(color=tuple(np.array(pal[labels])/255), label=class_dict[labels]))  
        
        fig,ax = plt.subplots(figsize =(22,30))
        plt.imshow(img2)
        plt.tight_layout()
        ax.legend(handles=handles, prop={'size': 20})
        ax.axis('off')
        img2 = img2.astype('uint8')
        fig.savefig(save_dir_noconf / f'{image_id}_annotated.jpeg')
        plt.close(fig)

        plt.figure(figsize =(30,20))
        plt.imshow(img)
        img = img.astype('uint8')
        plt.imsave(save_dir / f'{image_id}_annotated_conf.jpeg', img)
        plt.close()

    return str(save_dir)

