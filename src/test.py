import os
import standard_test
import sahi_test
import test_both
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TEST_DATA_PATH = "/home/sai/drone_ws/src/object_detect/data/VisImages/VisDrone2019-DET-test-dev/images"
# YAML_PATH = "/home/sai/drone_ws/src/object_detect/cfg/VisDrone.yaml"
# UNIFIED_RESULTS_PATH = "/home/sai/drone_ws/src/object_detect/runs/detect/test_output"
# COCO_JSON = "/home/sai/drone_ws/src/object_detect/cfg/coco_annotations.json"

# MODELS_CONFIG = {
#     # 'Yolov8': '/home/sai/drone_ws/src/object_detect/runs/detect/yolov8_800_16_200_coslr_dropout05/weights/best.pt',
#     # 'Yolov11': '/home/sai/drone_ws/src/object_detect/runs/detect/yolo11_800_16_200_coslr_dropout05/weights/best.pt',
#     'RT-DETR': '/home/sai/drone_ws/src/object_detect/runs/detect/rtdetr_800_16_200_coslr_dropout05/weights/best.pt'
# }

# # Ensure the unified results directory exists
# os.makedirs(UNIFIED_RESULTS_PATH, exist_ok=True)

# results_data = {
#     'Model': [],
#     'mAP50': [],
#     'mAP50-95': [],
#     'Average Precision': [],
#     'Average Recall': []
# }

# test_both.test(MODELS_CONFIG,
#                COCO_JSON,
#                TEST_DATA_PATH, 
#                results_data, 
#                UNIFIED_RESULTS_PATH, 
#                save_image = False)

