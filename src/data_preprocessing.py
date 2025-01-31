import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import argparse

classes_ = {}

def combine_classes(cls):
    # Combine pedestrian and people into one class
    if cls in [0, 1]:  # pedestrian or people
        return 0  # people
    elif cls in [2]:  # bicycle only
        return 1  # bicycle
    elif cls in [3, 4]:  # car or van
        return 2  # car
    elif cls in [5]:  # truck
        return 3  # truck
    elif cls in [8]:  # bus
        return 4  # bus
    elif cls in [9]:  # motor
        return 5  # motor
    
    return None

def visdrone2yolo(dir):
    from PIL import Image
    

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    Path(os.path.join(dir, 'labels')).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = Path(os.path.join(dir, 'annotations')).glob('*.txt')
    pbar = [files for files in pbar]
    for f in pbar:
        img_size = Image.open(Path(os.path.join(dir, 'images', f.name)).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue

                cls = int(row[5]) - 1
                cls = combine_classes(cls)

                if cls is None: 
                    continue

                if cls not in classes_:
                    classes_[cls] = 1
                else:
                    classes_[cls] += 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)
    print(f"Classes: {classes_}")

def convert2yolo(data_path, datasets):
    for dataset in datasets:
        visdrone2yolo(os.path.join(data_path, dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO labels')
    parser.add_argument('--data_path', type=str, default='VisImages',
                        help='Path to the VisDrone data directory')
    args = parser.parse_args()
    convert2yolo(args.data_path, ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev'])