import cv2
from ultralytics import YOLO, RTDETR
import time 

# Load the YOLO model
model = YOLO("/home/sai/drone_ws/src/visdrone_object_detection/runs/detect/yolo11_800_16_200_coslr_dropout05/weights/best.pt")

# Open the video file
video_path = "/home/sai/drone_ws/src/visdrone_object_detection/test_vids/aerial.mp4"
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

        # Display the annotated frame
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