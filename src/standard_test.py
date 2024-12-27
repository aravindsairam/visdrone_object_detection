from ultralytics import YOLO, RTDETR
import os
import cv2
import pandas as pd
import numpy as np


def get_image_result_path(UNIFIED_RESULTS_PATH, img_name):
    base_name = os.path.splitext(img_name)[0]
    img_result_path = os.path.join(UNIFIED_RESULTS_PATH, base_name)
    os.makedirs(img_result_path, exist_ok=True)
    return img_result_path


def draw_boxes(model, images, results):
    output_images = []
    for img, result in zip(images, results):
        for box in result.boxes:
            b = box.xyxy[0]
            c = box.cls
            conf = box.conf.item()  # Convert Tensor to float
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            label = f"{model.names[int(c)]} {conf:.2f}"  # Include class name and confidence
            cv2.putText(img, label, (int(b[0]), int(b[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        output_images.append(img)
    return output_images


def draw_ground_truth(image: np.ndarray, label_path: str) -> np.ndarray:
    """Draw ground truth annotations on the image."""
    if not os.path.exists(label_path):
        return image  # Return original image if no label file exists

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            img_h, img_w = image.shape[:2]

            # Convert from normalized to absolute coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"GT {class_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image


def test(MODELS_CONFIG, TEST_DATA_PATH, YAML_PATH, UNIFIED_RESULTS_PATH, results_data, save_image, batch_size=16):
    labels_path = "/home/sai/drone_ws/src/object_detect/data/VisImages/VisDrone2019-DET-test-dev/labels"
    
    for name, model_path in MODELS_CONFIG.items():
        if name == 'RT-DETR':
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
        
        if save_image:
            # Collect all image paths
            image_paths = [
                os.path.join(TEST_DATA_PATH, img_name)
                for img_name in os.listdir(TEST_DATA_PATH)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            # Process images in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = [cv2.imread(img_path) for img_path in batch_paths]
                results = model(batch_paths, conf=0.30)

                # Draw boxes and save images
                output_images = draw_boxes(model, batch_images, results)
                for img_path, img_with_boxes, result in zip(batch_paths, output_images, results):
                    img_name = os.path.basename(img_path)
                    img_result_path = get_image_result_path(UNIFIED_RESULTS_PATH, img_name)
                    
                    # Save predicted image
                    cv2.imwrite(os.path.join(img_result_path, f"{name}_result_{img_name}"), img_with_boxes)

                    # Save ground truth image
                    label_file = os.path.join(labels_path, f"{os.path.splitext(img_name)[0]}.txt")
                    ground_truth_image = cv2.imread(img_path).copy()
                    gt_with_boxes = draw_ground_truth(ground_truth_image, label_file)
                    cv2.imwrite(os.path.join(img_result_path, f"ground_truth_{img_name}"), gt_with_boxes)

                    # Save detection results to a file
                    with open(os.path.join(img_result_path, f"{name}_detections.txt"), "w") as f:
                        f.write(f"{name} Detections for {img_name}:\n")
                        for box in result.boxes:
                            b = box.xyxy[0]
                            conf = box.conf.item()
                            cls = int(box.cls.item())
                            f.write(f"  - Class: {model.names[cls]}, Confidence: {conf:.2f}, "
                                    f"Box: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}]\n")

        # Evaluate metrics
        metrics = model.val(data=YAML_PATH, split='test', conf = 0.30)
        results_data['Model'].append(name)
        results_data['mAP50'].append(metrics.results_dict.get('metrics/mAP50(B)', np.nan))
        results_data['mAP50-95'].append(metrics.results_dict.get('metrics/mAP50-95(B)', np.nan))
        results_data['Average Precision'].append(metrics.results_dict.get('metrics/precision(B)', np.nan))
        results_data['Average Recall'].append(metrics.results_dict.get('metrics/recall(B)', np.nan))

    # Save results to a CSV file
    results_table = pd.DataFrame(results_data)
    results_table.to_csv('visdrone_model_evaluation.csv', index=False)
    print(results_table)

    print(f"YOLO testing completed. Results saved in {UNIFIED_RESULTS_PATH}")