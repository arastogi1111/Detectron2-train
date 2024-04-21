import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.configurer import get_thing_classes

def perclass_AP(traineddir):
    
    traineddir = Path(traineddir)
    metrics_df = pd.read_json(traineddir / "metrics.json", orient="records", lines=True)
    mdf = metrics_df.sort_values("iteration")

    mdf3 = mdf[~mdf["bbox/AP75"].isna()]
    mdf_bbox_class = mdf3.iloc[-1][[f"bbox/AP-{col}" for col in get_thing_classes()]]

    mdf_bbox_class = pd.DataFrame(mdf_bbox_class)
    mdf_bbox_class.index = pd.Series(get_thing_classes())
    mdf_bbox_class.index.name = "class"
    
    mdf_bbox_class.columns = [ "bbox_AP"]
    mdf_bbox_class.to_csv(traineddir/"perclass_AP.csv")



