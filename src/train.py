from ultralytics import YOLO, RTDETR
import os

class model:
    def __init__(self, config_path):
        self.config_path = config_path

    def train(self, model_type, epochs, batch_size, imgsz, name):
        # Load a pretrained model
        if model_type == "yolo11":
            model = YOLO("yolo11s.pt")
        elif model_type == "RTDETR":
            model = RTDETR("rtdetr-l.pt")
        else:
            model = YOLO("yolov8s.pt")

        # Train the model
        results = model.train(data=os.path.join(self.config_path,"VisDrone.yaml"), epochs=epochs,
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
        
    def test(self, model_path, video_path):
        # Load a pretrained model
        model = RTDETR(model_path)

        # Train the model
        results = model(video_path, show=True, save=True)


if __name__ == "__main__":

    # training
    cfg = "/home/sai/drone_ws/src/object_detect/cfg"
    object_detection_yolov8 = model(cfg)
    object_detection_yolov8.train("RTDETR", epochs = 400,
                            batch_size = 4,
                            imgsz = 800,
                            name = "rtdetr_800_16_200_coslr_dropout05")
    
    # object_detection_yolov11 = model(cfg)
    # object_detection_yolov11.train("yolo11", epochs = 400,
    #                         batch_size = 16,
    #                         imgsz = 800,
    #                         name = "yolov11_800_16_400_coslr_dropout05")

    # testing 
    # test_model_path=  "/home/sai/drone_ws/src/object_detect/runs/detect/rtdetr_800_16_200_coslr_dropout05/weights/best.pt"
    # video_path = "/home/sai/drone_ws/src/object_detect/test_vids/aerial.mp4"
    # object_detection.test(test_model_path, video_path)




# run/train19 is the best for images 
# run/train20 is the best for images(people/pedestrians combined)
# run/train34 is the RTDETR for images(all class combined)
# run/train36 is the yolo11 for images(all class combined)
# run/train37 is the yolo8 for images(all class combined)
# run/train38 is the yolo8 for images(all class combined)/cosine lr/imgsz=720/batch=16/epochs=200