# Detectron 2 for Training on Pathologies
Module for utilizing [Detectron 2](https://github.com/facebookresearch/detectron2) to train models on radiological images.

#### Installations
 - [Requirements](requirements) to recreate environment.
 - Detectron2 0.4 ([version relevant](Detectron2_train/detectron2_train.py) to your Cuda version and pytorch version)
 
#### Dataset preparation
 - Data should be present as processed .png's in chosen directories specified in flags.
 - [Example](https://www.kaggle.com/xhlulu/vinbigdata) of data directory structure and form.
 - Include test_meta.csv if you wish to predict on testset.
 
#### Basic Script Preparation
 - [flags_dict](Detectron2_train/detectron2_train.py#L24) has temporarily been shifted to the [main script](Detectron2_train/detectron2_train.py)
 - Any other changes to be made to Detectron 2 config should be added to [Flags class](Detectron2_train/config/flags.py#L8) first.


### Training
 ```
 python Detectron2_train/detectron2_train.py
 ```

 
#### Notes
 - Some linux servers may require switching off torch.cuda.synchronize calls in [Loss Hook](Detectron2_train/model/loss_hook.py#L56) and Detectron2 [Evaluator](https://github.com/facebookresearch/detectron2/blob/61457a0178939ec8f7ce130fcb733a5a5d47df9f/detectron2/evaluation/evaluator.py#L159)
 - Change Device flag and directories. 


### Outputs
 - If left uncommented, after training is completed, metrics.json from output directory will be used to generate loss curves and AP values if validation set is present.
 - Predictions on test set in requisite Kaggle submissions format will automatically be generated.
 - From the formed csv submission, annotations will be drawn on sample with confidence bounds on BBs.
