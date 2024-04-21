
import pandas as pd
from pathlib import Path
import os
from datetime import datetime as dt
from datetime import timedelta as td

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from data.dataloaders import get_vinbigdata_dicts_QA, get_vinbigdata_dicts_test
from utils.flags_util import load_flags
from config.configurer import get_final_config
from utils.classes_ids import get_thing_classes
from testing.predictor import predict_over_dataset
from testing.filter2class import filter_2cls

def generate_submissions(trained_dir , csv_2class_preds="", pred_thr = 0.0, apply_2class_filter = True, low_thr  = 0.08, high_thr = 0.95 , use_kaggle_2CP = False):

    trained_dir = Path(trained_dir)
    flags = load_flags(trained_dir)
    inputdir = Path(flags.inputdir)
    imgdir = inputdir / flags.imgdir_name

    dataframe_dir = inputdir / flags.dataframe_dir

    debug = flags.debug
    thing_classes = get_thing_classes()
    if flags.use_class14:
        thing_classes.append("No finding")
        
    test_meta = pd.read_csv(dataframe_dir/"test_meta.csv")
    DatasetCatalog.register("vinbigdata_test", lambda: get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug))


    cfg = get_final_config(flags)
    cfg.MODEL.WEIGHTS = str(trained_dir/"model_final.pth")

    # set a custom testing threshold  : 0.0 for outputting all BB with any confidences at all.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = pred_thr  
    
    predictor = DefaultPredictor(cfg)
    # Model is accessible here, predictor.model

    MetadataCatalog.get("vinbigdata_test").set(thing_classes=thing_classes)
    metadata = MetadataCatalog.get("vinbigdata_test")
    dataset_dicts = get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug)

    if debug:
        dataset_dicts = dataset_dicts[:100]

    results_list = predict_over_dataset(predictor, dataset_dicts, test_meta)

    now = str((dt.now() + td(hours=5, minutes=30)).strftime("D_%Y-%m-%d_T_%H.%M.%S"))   # capturing current time in India for logging experiments.
    now = now[0:now.find(":")]
    submission_dir = trained_dir / f"Submissions_{now}"
    os.makedirs(submission_dir, exist_ok = True)

    submission_det = pd.DataFrame(results_list, columns=['image_id', 'PredictionString'])
    save_path = submission_dir / f"{now}_submission_{len(thing_classes)}class_th={pred_thr}.csv"
    submission_det.to_csv(save_path, index=False)

    
    if apply_2class_filter:

        pred_14cls = pd.read_csv(submission_dir / f"{now}_submission_{len(thing_classes)}class_th={pred_thr}.csv")
        pred_2cls = pd.read_csv(csv_2class_preds)
        pred = pd.merge(pred_14cls, pred_2cls, on = 'image_id', how = 'left')
        sub = pred.apply(lambda x : filter_2cls(x, low_thr, high_thr), axis=1)

        save_path = submission_dir /  f"{now}_submission_{len(thing_classes)}C_2CF_Kagg={use_kaggle_2CP}_lo={low_thr}_hi={high_thr}.csv"
        
        sub[['image_id', 'PredictionString']] \
            .to_csv(save_path, index=False)

        return save_path

    else:
        return save_path


def generate_qa_preds(trained_dir , csv_2class_preds_QA_set="", pred_thr = 0.0, apply_2class_filter = True, low_thr  = 0.08, high_thr = 0.95 , use_kaggle_2CP = False):

    trained_dir = Path(trained_dir)
    flags = load_flags(trained_dir)
    inputdir = Path(flags.inputdir)
    imgdir = inputdir / flags.imgdir_name
    inputdir = Path(flags.inputdir)
    debug = flags.debug
    thing_classes = get_thing_classes()
    if flags.use_class14:
        thing_classes.append("No finding")
        
    dataframe_dir = inputdir / flags.dataframe_dir
    qa_meta = pd.read_csv(dataframe_dir/"QA_meta.csv")

    DatasetCatalog.register("vinbigdata_QA", lambda: get_vinbigdata_dicts_QA(imgdir, qa_meta, debug=debug))


    cfg = get_final_config(flags)
    cfg.MODEL.WEIGHTS = str(trained_dir/"model_final.pth")

    # set a custom testing threshold  : 0.0 for outputting all BB with any confidences at all.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = pred_thr  
    
    predictor = DefaultPredictor(cfg)
    # Model is accessible here, predictor.model

    MetadataCatalog.get("vinbigdata_QA").set(thing_classes=thing_classes)
    metadata = MetadataCatalog.get("vinbigdata_QA")
    dataset_dicts = get_vinbigdata_dicts_QA(imgdir, qa_meta, debug=debug)

    

    results_list = predict_over_dataset(predictor, dataset_dicts, qa_meta)

    
    
    qa_dir = trained_dir / f"QA"
    os.makedirs(qa_dir, exist_ok = True)

    submission_det = pd.DataFrame(results_list, columns=['image_id', 'PredictionString'])
    save_path = qa_dir / f"QA_{len(thing_classes)}class_th={pred_thr}.csv"
    submission_det.to_csv(save_path, index=False)

    
    if apply_2class_filter:

        pred_14cls = pd.read_csv(qa_dir / f"QA_{len(thing_classes)}class_th={pred_thr}.csv")
        
        pred_2cls = pd.read_csv(csv_2class_preds_QA_set)

        pred = pd.merge(pred_14cls, pred_2cls, on = 'image_id', how = 'left')
        sub = pred.apply(lambda x : filter_2cls(x, low_thr, high_thr), axis=1)

        save_path = qa_dir /  f"QA_{len(thing_classes)}C_lo={low_thr}_hi={high_thr}.csv"
        
        sub[['image_id', 'PredictionString']] \
            .to_csv(save_path, index=False)

        return save_path

    else:
        return save_path

