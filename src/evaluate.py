import os
import generate_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Evaluate object detection models')
parser.add_argument('--test_data', type=str, 
                    default="VisImages/VisDrone2019-DET-test-dev/images",
                    help='Path to test data images')
parser.add_argument('--yaml_path', type=str,
                    default="cfg/VisDrone.yaml",
                    help='Path to YAML configuration file')
parser.add_argument('--results_path', type=str,
                    default="runs/detect/test_output",
                    help='Path to save unified results')
parser.add_argument('--coco_json', type=str,
                    default="cfg/coco_annotations.json",
                    help='Path to COCO annotations JSON file')
parser.add_argument('--yolov8_weights', type=str,
                    default="runs/detect/yolov8_800_16_200_coslr_dropout05/weights/best.pt",
                    help='Path to YOLOv8 model weights')
parser.add_argument('--yolov11_weights', type=str,
                    default="runs/detect/yolo11_800_16_200_coslr_dropout05/weights/best.pt",
                    help='Path to YOLOv11 model weights')
parser.add_argument('--rtdetr_weights', type=str,
                    default="runs/detect/rtdetr_800_16_200_coslr_dropout05/weights/best.pt",
                    help='Path to RT-DETR model weights')

args = parser.parse_args()


# Use parsed arguments
TEST_DATA_PATH = args.test_data
YAML_PATH = args.yaml_path
UNIFIED_RESULTS_PATH = args.results_path
COCO_JSON = args.coco_json

MODELS_CONFIG = {
    'Yolov8': args.yolov8_weights,
    'Yolov11': args.yolov11_weights,
    'RT-DETR': args.rtdetr_weights
}

# Ensure the unified results directory exists
os.makedirs(UNIFIED_RESULTS_PATH, exist_ok=True)

results_data = {
    'Model': [],
    'mAP50': [],
    'mAP50-95': [],
    'Average Precision': [],
    'Average Recall': []
}

generate_results.test(MODELS_CONFIG,
               COCO_JSON,
               TEST_DATA_PATH, 
               results_data, 
               UNIFIED_RESULTS_PATH, 
               save_image = False)

