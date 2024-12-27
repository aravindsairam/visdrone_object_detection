from ultralytics import YOLO
import os
import json
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image, visualize_object_predictions
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi.utils.coco import Coco

# best overlap_height_ratio = 0.2/0.3, slice_width = 720, confidence_threshold = 0.3

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_image_result_path(UNIFIED_RESULTS_PATH, img_name):
    base_name = os.path.splitext(img_name)[0]
    img_result_path = os.path.join(UNIFIED_RESULTS_PATH, base_name)
    os.makedirs(img_result_path, exist_ok=True)
    return img_result_path


def process_batch(
    batch_paths, batch_infos, predictions, detection_model, name, UNIFIED_RESULTS_PATH, coco, save_image
):
    category_to_id = {category.name: category.id for category in coco.categories}
    batch_images = [read_image(img_path) for img_path in batch_paths]
    
    # Perform batch predictions
    batch_results = [
        get_sliced_prediction(
            image,
            detection_model,
            slice_height=720,
            slice_width=720,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type = "NMS",
            postprocess_match_metric = "IOU",
            postprocess_match_threshold = 0.8,
        )
        for image in batch_images
    ]

    for image_path, image_info, result in zip(batch_paths, batch_infos, batch_results):
        img_name = os.path.basename(image_path)
        img_result_path = get_image_result_path(UNIFIED_RESULTS_PATH, img_name)
        
        for pred in result.object_prediction_list:
            predictions.append(
                {
                    'image_id': image_info.id,
                    'category_id': category_to_id[pred.category.name],
                    'bbox': [
                        pred.bbox.minx,
                        pred.bbox.miny,
                        pred.bbox.maxx - pred.bbox.minx,
                        pred.bbox.maxy - pred.bbox.miny,
                    ],
                    'score': pred.score.value,
                },
            )

        # Save SAHI visualization
        if save_image:
            sahi_vis = visualize_object_predictions(
                image=read_image(image_path),
                object_prediction_list=result.object_prediction_list,
            )

            if isinstance(sahi_vis, dict) and 'image' in sahi_vis:
                image_data = sahi_vis['image']
            elif isinstance(sahi_vis, np.ndarray):
                image_data = sahi_vis
            else:
                print(f"Unexpected type for visualization: {type(sahi_vis)} for {img_name}")
                continue

            cv2.imwrite(os.path.join(img_result_path, f"{name}_sahi_result_{img_name}"), image_data)

            with open(os.path.join(img_result_path, f"{name}_sahi_detections.txt"), "w") as f:
                f.write(f"{name} SAHI Detections for {img_name}:\n")
                for prediction in result.object_prediction_list:
                    f.write(f"  - Category: {prediction.category.name}, "
                            f"Score: {prediction.score.value:.2f}\n")


def test(MODELS_CONFIG, COCO_JSON, TEST_DATA_PATH, results_data, UNIFIED_RESULTS_PATH, save_image, batch_size=16):
    coco = Coco.from_coco_dict_or_path(COCO_JSON)
    pycoco = COCO(COCO_JSON)

    for name, model_path in MODELS_CONFIG.items():
        predictions = []
        if name == 'RT-DETR':
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="rtdetr",
                model_path=model_path,
                confidence_threshold=0.3,
                device="cuda:0",
                # agnostic_nms=True,
            )
        else:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=0.3,
                device="cuda:0",
                # agnostic_nms=True,
            )

        # Collect all image paths and infos
        image_infos = coco.images
        image_paths = [os.path.join(TEST_DATA_PATH, image_info.file_name) for image_info in image_infos]

        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_infos = image_infos[i:i+batch_size]
            process_batch(batch_paths, batch_infos, predictions, detection_model, name, UNIFIED_RESULTS_PATH, coco, save_image)

        # Save predictions
        predictions_path = f"{name}_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, cls=NumpyFloatValuesEncoder)

        # Load the predictions and evaluate
        pycoco_pred = pycoco.loadRes(predictions_path)
        coco_eval = COCOeval(pycoco, pycoco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results_data['Model'].append(name)
        results_data['mAP50'].append(np.mean(coco_eval.eval['precision'][0, :, :, 0, 2]))
        results_data['mAP50-95'].append(np.mean(coco_eval.eval['precision'][0, :, :, 0, :]))
        results_data['Average Precision'].append(np.mean(coco_eval.eval['precision'][:, :, :, 0, -1]))
        results_data['Average Recall'].append(np.mean(coco_eval.eval['recall'][:, :, 0, -1]))

    # Save results to CSV
    results_table = pd.DataFrame(results_data)
    results_table.to_csv('visdrone_sahi_model_evaluation.csv', index=False)
    print(results_table)

    print(f"SAHI testing completed. Results saved in {UNIFIED_RESULTS_PATH}")