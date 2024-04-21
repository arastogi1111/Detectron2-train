
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('max_columns', 50)


# ----  Our Detectron2_train Module imports  ----

from data.dataloaders import get_vinbigdata_dicts
from utils.classes_ids import get_thing_classes


from detectron2.data import DatasetCatalog, MetadataCatalog




def split_and_register(flags):

    # --- Read data ---
    inputdir = Path(flags.inputdir)
    dataframe_dir = inputdir / flags.dataframe_dir
    imgdir = inputdir / flags.imgdir_name

    # Read in the data CSV files
    train_df = pd.read_csv( dataframe_dir / "train.csv")
    # train = train_df  # alias
    # sample_submission = pd.read_csv(datadir / 'sample_submission.csv')

    thing_classes = get_thing_classes()
    debug = flags.debug

    train_data_type = flags.train_data_type

    if flags.use_class14:
        thing_classes.append("No finding")

    split_mode = flags.split_mode
    
    
    if split_mode == "all_train":
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
    
    
    elif split_mode == "valid20":
        # To get number of data...
        n_dataset = len(
            get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            )
        )
        n_train = int(n_dataset * 0.8)
        print("n_dataset", n_dataset, "n_train", n_train)
        rs = np.random.RandomState(flags.seed)
        inds = rs.permutation(n_dataset)
        train_inds, valid_inds = inds[:n_train], inds[n_train:]
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=train_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
        DatasetCatalog.register(
            "vinbigdata_valid",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=valid_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_valid").set(thing_classes=thing_classes)

    elif split_mode == "all_minus_qa":
        
        qa_list = pd.read_csv(dataframe_dir / "QA_list.csv")

        train_minus_qa_df = train_df[~train_df.image_id.isin(qa_list.image_id)]

        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir, train_minus_qa_df, train_data_type, debug=debug, use_class14=flags.use_class14
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)

    elif split_mode == "val_minus_qa":
        
        VAL_CSV = dataframe_dir / "VAL_Files_IDs.csv"
        val_ids = pd.read_csv(VAL_CSV).Image_ID

        qa_list = pd.read_csv(dataframe_dir / "QA_list.csv")

        val_df = train_df[train_df.image_id.isin(val_ids)]
        train_minus_val_df = train_df[~train_df.image_id.isin(val_ids)]
        
        val_minus_qa = val_df[~val_df.image_id.isin(qa_list.image_id)]

        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_minus_val_df,
                train_data_type,
                debug=debug,
                # target_indices=train_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)

        DatasetCatalog.register(
            "vinbigdata_valid",
            lambda: get_vinbigdata_dicts(
                imgdir,
                val_minus_qa,
                train_data_type,
                debug=debug,
                # target_indices=valid_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_valid").set(thing_classes=thing_classes)
        

    else:
        raise ValueError(f"[ERROR] Unexpected value split_mode={split_mode}")



    # dataset_dicts = get_vinbigdata_dicts(imgdir, train, debug=debug)

