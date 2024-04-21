import cv2
from tqdm import tqdm
from math import ceil
from typing import Any, Dict, List

from testing.format_predict import format_pred, predict_batch

def predict_over_dataset(predictor, dataset_dicts, test_meta):
    results_list = []
    index = 0
    batch_size = 4

    for i in tqdm(range(ceil(len(dataset_dicts) / batch_size))):
        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dicts))))
        dataset_dicts_batch = [dataset_dicts[i] for i in inds]
        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        outputs_list = predict_batch(predictor, im_list)

        for im, outputs, d in zip(im_list, outputs_list, dataset_dicts_batch):
            resized_height, resized_width, ch = im.shape
            # outputs = predictor(im)
            # if index < 5:
            #     # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            #     v = Visualizer(
            #         im[:, :, ::-1],
            #         metadata=metadata,
            #         scale=0.5,
            #         instance_mode=ColorMode.IMAGE_BW
            #         # remove the colors of unsegmented pixels. This option is only available for segmentation models
            #     )
            #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            #     # cv2_imshow(out.get_image()[:, :, ::-1])
            #     cv2.imwrite(str(outdir / f"pred_{index}.jpg"), out.get_image()[:, :, ::-1])
            # breakpoint()
            image_id, dim0, dim1 = test_meta.iloc[index].values

            instances = outputs["instances"]
            if len(instances) == 0:
                # No finding, let's set 14 1 0 0 1 1x.
                result = {"image_id": image_id, "PredictionString": "14 1.0 0 0 1 1"}
            else:
                # Find some bbox...
                # print(f"index={index}, find {len(instances)} bbox.")
                fields: Dict[str, Any] = instances.get_fields()
                pred_classes = fields["pred_classes"]  # (n_boxes,)
                pred_scores = fields["scores"]
                # shape (n_boxes, 4). (xmin, ymin, xmax, ymax)
                pred_boxes = fields["pred_boxes"].tensor

                h_ratio = dim0 / resized_height
                w_ratio = dim1 / resized_width
                pred_boxes[:, [0, 2]] *= w_ratio
                pred_boxes[:, [1, 3]] *= h_ratio

                pred_classes_array = pred_classes.cpu().numpy()
                pred_boxes_array = pred_boxes.cpu().numpy()
                pred_scores_array = pred_scores.cpu().numpy()

                result = {
                    "image_id": image_id,
                    "PredictionString": format_pred(
                        pred_classes_array, pred_boxes_array, pred_scores_array
                    ),
                }
            results_list.append(result)
            index += 1

    
    return results_list