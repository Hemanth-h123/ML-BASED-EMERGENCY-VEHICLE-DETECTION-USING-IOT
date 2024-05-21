import cv2
from ultralytics import YOLO
import os
import torch
import matplotlib.pyplot as plt
import librosa
import numpy as np
from tensorflow import keras
import serial
ser = serial.Serial('COM4', baudrate=9600, timeout=1)
org = (20, 100)
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)  # Red
font_thickness = 1
line_type = cv2.LINE_AA


def extract_sound_from_frame(frame):
    audio_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sample_rate = 44100
    audio_data = np.array(audio_data.tolist())
    print("********",audio_data.shape)
    
    audio_features = np.mean(audio_data.reshape(1,80,-1),axis=-1)
    audio_features = audio_features.reshape(1, 80, 1)
    
    return audio_features

def predict_label_from_audio_features(audio_features):
    # Use your audio model for prediction
    y_pred = audio_model.predict(audio_features)
    y_pred = max(y_pred[0])
    print("Audio Result: ",y_pred)
    y_pred = int(np.median(y_pred))
    print("Audio Result: ",y_pred)

    return y_pred


# Load the YOLOv8 model
yolo_model = YOLO('runs/detect/Emergency_yolov8s_100s/weights/best.pt',task='detect')

audio_model = keras.models.load_model("files/Emergency_vehicle_model.h5")

video_file = "files/video (2160p).mp4" 
cap = cv2.VideoCapture(video_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    yolo_results = yolo_model.predict(frame, conf=0.50)
    Count = len(yolo_results[0])

    # Audio detection
    audio_features = extract_sound_from_frame(frame)
    print("********",audio_features.shape)
    audio_pred = predict_label_from_audio_features(audio_features)

    annotated_frame = yolo_results[0].plot()
    cv2.putText(annotated_frame, f"Object Count: {Count}", org, font_face, font_scale, font_color, font_thickness, line_type)
    
    if audio_pred == 1:
        ser.write(b'1')
        cv2.putText(annotated_frame, "Emergency Vehicle Detected", (20, 150), font_face, font_scale, font_color, font_thickness, line_type)
    elif audio_pred == 0:
        cv2.putText(annotated_frame, "Traffic Sound Detected", (20, 150), font_face, font_scale, font_color, font_thickness, line_type)

    cv2.imshow("Object and Audio Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
