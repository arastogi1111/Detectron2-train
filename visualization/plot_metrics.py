import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)

from config.configurer import get_thing_classes

def plot_metrics(traineddir , split : str):
    
    traineddir = Path(traineddir)
    metrics_df = pd.read_json(traineddir / "metrics.json", orient="records", lines=True)
    mdf = metrics_df.sort_values("iteration")


    fig, ax = plt.subplots()

    mdf1 = mdf[~mdf["total_loss"].isna()]
    ax.plot(mdf1["iteration"], mdf1["total_loss"], c="C0", label="train")
    if "validation_loss" in mdf.columns:
        mdf2 = mdf[~mdf["validation_loss"].isna()]
        ax.plot(mdf2["iteration"], mdf2["validation_loss"], c="C1", label="validation")


    outdir = traineddir / "figures"
    os.makedirs(outdir, exist_ok=True)

    # ax.set_ylim([0, 0.5])
    ax.legend()
    ax.set_title("Loss curve")
    plt.show()
    plt.savefig(outdir/"loss.png")

    if split not in [ "all_train", "all_minus_qa" ]:
        
        fig, ax = plt.subplots()
        mdf3 = mdf[~mdf["bbox/AP75"].isna()]
        ax.plot(mdf3["iteration"], mdf3["bbox/AP75"] / 100., c="C2", label="validation")

        ax.legend()
        ax.set_title("AP40")
        plt.show()
        fig.savefig(outdir / "AP40.png")


        fig, ax = plt.subplots(figsize=(15,10))
        mdf_bbox_class = mdf3.iloc[-1][[f"bbox/AP-{col}" for col in get_thing_classes()]]
        mdf_bbox_class.plot(kind="bar", ax=ax)
        ax.set_title("AP by class")
        plt.tight_layout()
        fig.savefig(outdir / "AP_by_class.png")

