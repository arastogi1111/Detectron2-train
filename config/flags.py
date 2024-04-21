
# --- flags ---
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Flags:
    # General
    debug: bool = True
    inputdir: str = "/kaggle/input"
    outdir: str = "results/det"

    # Data config
    train_orig_dir: str = "vinbigdata-chest-xray-abnormalities-detection"
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    dataframe_dir: str = "dataframes"
    
    split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    train_data_type: str = "original"  # original or wbf
    use_class14: bool = False
    device : str = "cpu"
    
    config_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    download_weights: bool = False
    local_model_weights: str = "/app/703268403/projects/Dicom/kaggle/ChestCT/vinbigdata-chest-xray-abnormalities-detection/Checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    
    # Training config
    iter: int = 10000
    ims_per_batch: int = 2  # images per batch, this corresponds to "total batch size"
    num_workers: int = 4
    lr_scheduler_name: str = "WarmupMultiStepLR"  # WarmupMultiStepLR (default) or WarmupCosineLR
    base_lr: float = 0.00025
    roi_batch_size_per_image: int = 512
    eval_period: int = 10000
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self
