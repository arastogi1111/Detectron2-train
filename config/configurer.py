import os
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN

from utils.flags_util import save_flags, load_flags
from utils.classes_ids import get_thing_classes

  
def get_final_config(flags):
    
    if not flags: 
        flags = load_flags()
    
    split_mode = flags.split_mode

    cfg = get_cfg()
    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg

    
    original_output_dir = cfg.OUTPUT_DIR
    outdir = Path(flags.outdir)
    cfg.OUTPUT_DIR = str(outdir)
    print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = ("vinbigdata_train",)
    
    if split_mode in [ "all_train", "all_minus_qa" ] :
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ("vinbigdata_valid",)
        cfg.TEST.EVAL_PERIOD = flags.eval_period

    cfg.MODEL.DEVICE = flags.device  
    cfg.DATALOADER.NUM_WORKERS = flags.num_workers

    
    if flags.download_weights=="True":
        config_name = flags.config_name
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)    ## SSL VERIFICATION ERROR YET AGAIN.
    else:
        cfg.MODEL.WEIGHTS = flags.local_model_weights
    
    cfg.SOLVER.IMS_PER_BATCH = flags.ims_per_batch
    cfg.SOLVER.LR_SCHEDULER_NAME = flags.lr_scheduler_name
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000  # Small value=Frequent save need a lot of storage.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image

    thing_classes = get_thing_classes()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    # NOTE: this config means the number of classes,
    # but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg
