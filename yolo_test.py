# Import libraries
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (You can replace 'yolov8n' with 'yolov8s', 'yolov8m', etc., depending on your performance needs)
model = YOLO('yolov8n.pt') # 'n' is for nano, a smaller, faster model

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing if needed
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = model(frame) # YOLO model inference

    # Loop through the detections and draw bounding boxes
    for result in results[0].boxes: # Access the first item (image results) and boxes
        box = result.xyxy[0] # Get the box coordinates in xyxy format
        confidence = result.conf[0] # Confidence of the detection
        class_id = int(result.cls[0]) # Class ID of the detection
        label = model.names[class_id] # Get label name from class ID

        # Draw the bounding box
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, box)
        cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)

        # Add label and confidence score
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to break
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
