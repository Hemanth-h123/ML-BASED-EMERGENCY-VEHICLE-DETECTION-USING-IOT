import cv2
from ultralytics import YOLO
import os

# Define the font parameters
org = (20, 100)
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)  # Red
font_thickness = 1
line_type = cv2.LINE_AA

# Load the YOLOv8 model
model = YOLO('runs/detect/Emergency_yolov8s_100s/weights/best.pt')

# Open the video file
folder_path = "test_images"
folder = os.listdir("test_images")
try:
        folder.remove(".DS_Store")
except:
        pass

for i in folder:
        img_path = f'{folder_path}'+"/"+f'{i}'
        results = model.predict(img_path,conf=0.1)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        Count = len(results[0])

        cv2.putText(annotated_frame, f"Object Count:{Count}", org, font_face, font_scale, font_color, font_thickness, line_type)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        cv2.waitKey(50000)
cv2.destroyAllWindows()