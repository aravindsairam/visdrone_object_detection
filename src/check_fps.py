import cv2
from ultralytics import YOLO, RTDETR
import time 
import argparse

def check_fps(model_type, model_path, video_path):
    # Load the YOLO model
    if model_type == 'YOLO':
        model = YOLO(model_path)
    elif model_type == 'RTDETR':
        model = RTDETR(model_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    prev_time = 0
    fps = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # # Display the annotated frame
            cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YOLO object detection with FPS counter')
    parser.add_argument('--model_type', type=str,
                    choices=['YOLO', 'RTDETR'],
                    default='YOLO',
                    help='Type of model to use (YOLO or RTDETR)')
    parser.add_argument('--model', type=str, 
                        default="/home/sai/drone_ws/src/visdrone_object_detection/runs/detect/rtdetr_800_16_200_coslr_dropout05/weights/best.pt",
                        help='Path to YOLO model weights')
    parser.add_argument('--video', type=str, 
                        default="/home/sai/drone_ws/src/visdrone_object_detection/test_vids/aerial.mp4",
                        help='Path to input video file')
    
    args = parser.parse_args()
    check_fps(args.model_type, args.model, args.video)