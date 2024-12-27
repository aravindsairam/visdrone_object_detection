from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco
from typing import Any

from PIL import Image

class COCOConverter:
    """
    Converts YOLO format annotations to COCO format.
    """

    def __init__(self, categories: list[str]):
        """
        Initialises COCO data structure and category mappings.

        Args:
            categories (List[str]): A list of category names.
        """
        self.coco_format: dict[str, list[Any]] = {
            'images': [],
            'annotations': [],
            'categories': [],
        }
        self.initialise_categories(categories)
        self.image_id = 1  # Unique ID for each image
        self.annotation_id = 1  # Unique ID for each annotation

    def initialise_categories(self, categories: list[str]):
        """Initialises categories for COCO format.

        Args:
            categories (List[str]): A list of category names.
        """
        for i, category in enumerate(categories):
            self.coco_format['categories'].append(
                {
                    'id': i + 1,
                    'name': category,
                    'supercategory': 'none',
                },
            )

    def convert_annotations(self, labels_dir: str, images_dir: str):
        """Reads YOLO formatted annotations and converts them to COCO format.

        Args:
            labels_dir (str): Directory containing YOLO labels.
            images_dir (str): Directory containing image files.
        """
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                image_name = filename.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Warning: {image_path} does not exist.")
                    continue

                image = Image.open(image_path)
                width, height = image.size

                self.coco_format['images'].append(
                    {
                        'id': self.image_id,
                        'width': width,
                        'height': height,
                        'file_name': image_name,
                    },
                )

                with open(os.path.join(labels_dir, filename)) as file:
                    for line in file:
                        cls_id, x_center, y_center, bbox_width, bbox_height = (
                            (
                                float(x) if float(x) != int(
                                    float(x),
                                ) else int(float(x))
                            )
                            for x in line.strip().split()
                        )

                        x_min = (x_center - bbox_width / 2) * width
                        y_min = (y_center - bbox_height / 2) * height
                        bbox_width *= width
                        bbox_height *= height

                        self.coco_format['annotations'].append(
                            {
                                'id': self.annotation_id,
                                'image_id': self.image_id,
                                'category_id': int(cls_id) + 1,
                                'bbox': [
                                    x_min, y_min, bbox_width, bbox_height,
                                ],
                                'area': bbox_width * bbox_height,
                                'segmentation': [],
                                'iscrowd': 0,
                            },
                        )
                        self.annotation_id += 1
                self.image_id += 1

    def save_to_json(self, output_path: str):
        """Saves the COCO formatted data to a JSON file.

        Args:
            output_path (str): Path to save the JSON output.
        """
        with open(output_path, 'w') as json_file:
            json.dump(self.coco_format, json_file, indent=4)

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
class COCOEvaluator:
    """
    Evaluates object detection models using COCO metrics.
    """

    def __init__(
        self,
        model_path: str,
        coco_json: str,
        image_dir: str,
        confidence_threshold: float = 0.3,
        slice_height: int = 370,
        slice_width: int = 370,
        overlap_height_ratio: float = 0.3,
        overlap_width_ratio: float = 0.3,
    ):
        """
        Initialises the evaluator with model and dataset parameters.

        Args:
            model_path (str): Path to the trained model file.
            coco_json (str): Path to the COCO format annotations JSON file.
            image_dir (str): Directory containing the evaluation image set.
            confidence_threshold (float, optional): Threshold for preds.
                Defaults to 0.3.
            slice_height (int, optional): Height of the slices for pred.
                Defaults to 370.
            slice_width (int, optional): Width of the slices for pred.
                Defaults to 370.
            overlap_height_ratio (float, optional): Height slice overlap ratio.
                Defaults to 0.3.
            overlap_width_ratio (float, optional): Width slice overlap ratio.
                Defaults to 0.3.
        """
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov11',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            # device="cpu",  # Uncomment this to force CPU usage
        )
        self.coco_json = coco_json
        self.image_dir = image_dir
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def evaluate(self) -> dict[str, float]:
        """
        Evaluates the model on the dataset and computes COCO metrics.

        Returns:
            Dict[str, float]: A dictionary containing computed metrics.
        """
        print(f"Evaluating model with data path: {self.coco_json}")
        coco = Coco.from_coco_dict_or_path(self.coco_json)
        pycoco = COCO(self.coco_json)
        predictions = []
        category_to_id = {
            category.name: category.id for category in coco.categories
        }

        for image_info in coco.images:
            image_path = os.path.join(self.image_dir, image_info.file_name)
            print(f"Processing image: {image_path}")
            prediction_result = get_sliced_prediction(
                image_path,
                self.model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
            )
            for pred in prediction_result.object_prediction_list:
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

        # Save the predictions to a JSON file
        predictions_path = 'predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, cls=NumpyFloatValuesEncoder)

        # Load the predictions and evaluate
        pycoco_pred = pycoco.loadRes(predictions_path)
        coco_eval = COCOeval(pycoco, pycoco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            'Average Precision': np.mean(
                coco_eval.eval['precision'][:, :, :, 0, -1],
            ),
            'Average Recall': np.mean(
                coco_eval.eval['recall'][:, :, 0, -1],
            ),
            'mAP at IoU=50': np.mean(
                coco_eval.eval['precision'][0, :, :, 0, 2],
            ),
            'mAP at IoU=50-95': np.mean(
                coco_eval.eval['precision'][0, :, :, 0, :],
            ),
        }
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluates a YOLO model using COCO metrics.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model file.',
    )
    parser.add_argument(
        '--coco_json',
        type=str,
        required=True,
        help='Path to the COCO format annotations JSON file.',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing the evaluation image set.',
    )
    args = parser.parse_args()
    evaluator = COCOEvaluator(
        model_path=args.model_path,
        coco_json=args.coco_json,
        image_dir=args.image_dir,
    )
    metrics = evaluator.evaluate()
    print('Evaluation metrics:', metrics)


if __name__ == '__main__':
    main()

"""example usage
python sahi_eval.py \
    --model_path "../../models/pt/best_yolov8x.pt" \
    --coco_json "dataset/coco_annotations.json" \
    --image_dir "dataset/valid/images"
"""