from ultralytics import YOLO, RTDETR
import os
import argparse

def train(config_path, model_type, epochs, batch_size, imgsz, name):
    # Load a pretrained model
    if model_type == "yolo11":
        model = YOLO("yolo11s.pt")
    elif model_type == "RTDETR":
        model = RTDETR("rtdetr-l.pt")
    else:
        model = YOLO("yolov8s.pt")

    # Train the model
    results = model.train(data=os.path.join(config_path, "VisDrone.yaml"), epochs=epochs,
                                            batch=batch_size,
                                            imgsz=imgsz,
                                            close_mosaic=50,
                                            shear=0.1,
                                            bgr=0.1, 
                                            mixup=0.1,
                                            copy_paste=0.1, 
                                            patience=150, 
                                            cos_lr=True,
                                            dropout = 0.5,
                                            name = name)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config_path', type=str, default='cfg', help='Path to the config file')
    parser.add_argument('--model_type', type=str, default='RTDETR', help='Model type')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=800, help='Image size')
    parser.add_argument('--name', type=str, default='run_1', help='Name of the run')
    args = parser.parse_args()

    # training
    train(args.config_path, args.model_type, args.epochs, args.batch_size, args.imgsz, args.name)