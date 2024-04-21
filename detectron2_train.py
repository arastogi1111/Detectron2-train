from pathlib import Path
import torch
from shutil import make_archive

# Our Detectron2_train Module imports 
from model.trainer import MyTrainer
from utils.flags_util import save_flags, load_flags
from utils.summarizer import vin_summarize
from config.configurer import get_final_config
from visualization.plot_metrics import plot_metrics
from data.split_and_register import split_and_register
from testing.generate_submissions import generate_qa_preds, generate_submissions
from testing.extract_metrics import perclass_AP
from visualization.visualize import visualize_QA_preds, visualize_submission_preds
from utils.kaggle import submit_to_kaggle

from detectron2.utils.logger import setup_logger
setup_logger()

from datetime import datetime as dt
from datetime import timedelta as td
now = str((dt.now() + td(hours=5, minutes=30)).strftime("D_%Y-%m-%d_T_%H.%M.%S"))   # capturing current time in India for logging experiments.
now = now[0:now.find(":")]


from pycocotools.cocoeval import COCOeval
print("HACKING: overriding COCOeval.summarize = vin_summarize...")
COCOeval.summarize = vin_summarize

use_kaggle_2CP = False

flags_dict = {
    "debug": False,
    "inputdir": "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/datasets",
    "dataframe_dir" : "dataframes",

    "outdir": f"/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/detectron2/14_classes_v1024/v5_{now}", 
    
    "imgdir_name": "v1024_png/vinbigdata",
    "device": "cuda:0",
    "split_mode": "all_minus_qa",


    "download_weights" : "False",
    "local_model_weights": "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
    
    "iter": 40000,
    "ims_per_batch": 4,
    "roi_batch_size_per_image": 512,
    "eval_period": 4000,
    "lr_scheduler_name": "WarmupCosineLR",
    "base_lr": 0.001,
    "num_workers": 4,
    "aug_kwargs": {
        "HorizontalFlip": {"p": 0.5},
        "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5}
    }
}

# Output csv of 2 class classifier output
if use_kaggle_2CP:
    # Kaggle output from https://www.kaggle.com/awsaf49/vinbigdata-2class-prediction
    csv_2class_preds = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/2_class/2-cls test pred.csv"
else:
    csv_2class_preds = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/detectron2/Sai/2_class/v1/test_pred_detect_SAI_29.7.21.csv"

csv_2class_preds_QA_set = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/2_class/QA_SET_pred_detect_but_trained_onit_2.8.21.csv"




if __name__ == "__main__":
    # Choice 1 : Use the flags in dict
    flags = save_flags(flags_dict)
    
    # # Choice 2 : Go to a previous output directory and get flags.
    # outdir = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/detectron2/14_classes_v1024/v5_D_2021-08-04_T_13.14.2"
    # flags = load_flags(outdir)
    
    split_and_register(flags)
    cfg = get_final_config(flags)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    outdir = Path(cfg.OUTPUT_DIR)
    
    if flags.split_mode not in [ "all_train", "all_minus_qa" ]:
        perclass_AP(outdir)
    plot_metrics(outdir, flags.split_mode)

    # # # Prediction on Test set and kaggle submissions generation
    # # # 2 class filter on by default, can pass thresholds in parameters
    sub_csv_path = generate_submissions(outdir , csv_2class_preds, 
                                            pred_thr = 0.0, apply_2class_filter = True, 
                                            low_thr  = 0.05, high_thr = 0.4,
                                            use_kaggle_2CP = use_kaggle_2CP)
    
    submit_to_kaggle(sub_csv_path , "v1024_All_minus_qa_40K_Akarsh_5.8.21_2CF_Sai")

    

    # # Draw BBs over some samples from test set, from predictions just made.
    visualize_submission_preds(sub_csv_path, samples = 20)


    # # QA SET predictions and viz

    qa_csv_path = generate_qa_preds(outdir, csv_2class_preds_QA_set= csv_2class_preds_QA_set, 
                                            pred_thr = 0.2, apply_2class_filter = True, 
                                            low_thr  = 0.07, high_thr = 0.4,
                                            use_kaggle_2CP = use_kaggle_2CP)
    
    save_dir = visualize_QA_preds(qa_csv_path, threshold = 0.3)
    make_archive(save_dir,'zip',save_dir)

    

    