import os
from pathlib import Path
import dataclasses

# ----  Our Detectron2_train Module imports  ----
from utils.yaml import save_yaml, load_yaml
from config.flags import Flags
# from config.flags_dict import get_flags_dict


# args = parse()
def save_flags(flags_dict):
    # flags_dict = get_flags_dict()
    flags = Flags().update(flags_dict)
    print("flags", str(flags))
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)
    flags_dict = dataclasses.asdict(flags)
    save_yaml(outdir / "flags.yaml", flags_dict)

    return flags


def load_flags(outdir):
    # flags_dict = get_flags_dict()
    if outdir is None:
        print("Provide output directory to find flags")

    outdir = Path(outdir)
    flags_dict = load_yaml(outdir / "flags.yaml")
    flags = Flags().update(flags_dict)
    
    return flags
    


