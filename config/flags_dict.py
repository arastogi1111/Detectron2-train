# from datetime import datetime as dt
# from datetime import timedelta as td
# now = str((dt.now() + td(hours=5, minutes=30)).strftime("D_%Y-%m-%d_T_%H.%M.%S"))   # capturing current time in India for logging experiments.
# now = now[0:now.find(":")]

# flags_dict = {
#     "debug": False,
#     "inputdir": "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/datasets",
#     "train_csv_orig_dir" : "original_dcm",
#     "outdir": f"/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Outputs/detectron2/14_classes_v512/v1_D_2021-07-16_T_04.28.4", 
#     "imgdir_name": "v512_png",
#     "device": "cuda:0",
#     "split_mode": "all_train",
#     "iter": 30000,
#     "ims_per_batch": 2,
#     "roi_batch_size_per_image": 512,
#     "eval_period": 2000,
#     "lr_scheduler_name": "WarmupCosineLR",
#     "base_lr": 0.001,
#     "num_workers": 4,
#     "aug_kwargs": {
#         "HorizontalFlip": {"p": 0.5},
#         "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
#         "RandomBrightnessContrast": {"p": 0.5}
#     }
# }

# def get_flags_dict():
#     return flags_dict