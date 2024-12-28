from ultralytics import YOLO, RTDETR
import os
import argparse

def test(model_type, model_path, video_path):
    # Load a pretrained model
    if model_type == "RTDETR":
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)

    # Train the model
    results = model(video_path, show=True, save=True)


# testing 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--model_type', type=str, default='RTDETR', help='Model type')
    parser.add_argument('--model_path', type=str, default='runs/detect/rtdetr/weights/best.pt', help='Path to the model')
    parser.add_argument('--video_path', type=str, default='aerial.mp4', help='Path to the video')
    args = parser.parse_args()
    test(args.model_type, args.model_path, args.video_path)
