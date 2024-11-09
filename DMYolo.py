from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import time
import numpy as np
import cv2
from ultralytics import YOLO

device = "cpu"
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

# Load the YOLOv8 model (You can replace 'yolov8n' with 'yolov8s', 'yolov8m', etc., depending on your performance needs)
model = YOLO('yolov8n.pt')  # 'n' is for nano, a smaller, faster model

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

i = 0


iteration = 0
while True:
    tup_arr = []
    start_time = time.time()
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame to half its original resolution
    # frame_small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    frame_small = cv2.resize(frame, (640//2, 480//2))

    # Convert the resized frame to RGB format for the pipeline
    image = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    depth = pipe(image)["depth"]
    depth_array = np.array(depth)

    # Normalize depth array between 0 and 1 for true depth calculation
    min_depth, max_depth = depth_array.min(), depth_array.max()
    depth_normalized = (depth_array - min_depth) / (max_depth - min_depth)

    # Calculate A and B for the true depth formula
    A = 1 / max_depth
    B = (1 / min_depth) - A

    # Compute true depth for each pixel
    true_depth_array = 1 / (A + B * depth_normalized)

    # Visualize the depth map
    depth_display = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)


    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save only the smaller depth image
    if i < 10:
        cv2.imwrite(f"depth_photos/0{i}.png", depth_colored)
        i += 1

    results = model(frame_small)
    for result in results[0].boxes:  # Access the first item (image results) and boxes
        box = result.xyxy[0]  # Get the box coordinates in xyxy format
        confidence = result.conf[0]  # Confidence of the detection
        class_id = int(result.cls[0])  # Class ID of the detection
        label = model.names[class_id]  # Get label name from class ID

        # Draw the bounding box
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, box)
        cv2.rectangle(depth_colored, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)

        # Add label and confidence score
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(depth_colored, label_text, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

        if(confidence > 0.4):
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, box)
            x_center = (x_top_left + x_bottom_right) // 2
            y_center = (y_top_left + y_bottom_right) // 2
            true_depth = depth_array[y_center, x_center]
            tup_arr.append((iteration, label, x_top_left, y_top_left, x_bottom_right, y_bottom_right, true_depth))
            # print(x_center, y_center, true_depth)

    # Display the original frame and depth map in their original size
    # cv2.imshow('Original', frame)
    cv2.imshow('Depth Map (Color)', depth_colored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    iteration += 1
    print(tup_arr)
cap.release()
cv2.destroyAllWindows()
