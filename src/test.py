from ultralytics import YOLO, RTDETR
import os


def test(self, model_path, video_path):
        # Load a pretrained model
        model = RTDETR(model_path)

        # Train the model
        results = model(video_path, show=True, save=True)


# testing 
    # test_model_path=  "/home/sai/drone_ws/src/object_detect/runs/detect/rtdetr_800_16_200_coslr_dropout05/weights/best.pt"
    # video_path = "/home/sai/drone_ws/src/object_detect/test_vids/aerial.mp4"
    # object_detection.test(test_model_path, video_path)