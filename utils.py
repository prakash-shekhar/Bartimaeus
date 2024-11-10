import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline, CLIPSegProcessor, CLIPSegForImageSegmentation
from ultralytics import YOLO
import os
import logging

def speak_mac(text):
    os.system(f"say '{text}'")

logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
yolo_model = YOLO('yolov8n.pt')

def run_semantic_segmentation(frame, prompt="chair", overlay_original=False):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    seg_inputs = seg_processor(text=[prompt], images=[pil_image], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        seg_outputs = seg_model(**seg_inputs)
    preds = seg_outputs.logits.unsqueeze(1)
    mask = torch.sigmoid(preds[0][0]).cpu().numpy()
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

    if overlay_original:
        overlay = frame.copy()
        overlay[:, :, 0] = np.where(binary_mask == 255, 0, 0)
        overlay[:, :, 1] = np.where(binary_mask == 255, 255, 0)
        overlay[:, :, 2] = np.where(binary_mask == 255, 0, 255)
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    return binary_mask

def run_object_detection(frame):
    detection_frame = frame.copy()
    yolo_results = yolo_model(detection_frame)
    for result in yolo_results[0].boxes:
        box = result.xyxy[0]
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        label = yolo_model.names[class_id]

        x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, box)
        cv2.rectangle(detection_frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(detection_frame, label_text, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return detection_frame

def run_depth_estimation(frame, overlay_detection=False):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth = depth_pipe(pil_image)["depth"]
    depth_array = np.array(depth)
    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

    if overlay_detection:
        detection_frame = run_object_detection(frame)
        for result in yolo_model(detection_frame)[0].boxes:
            box = result.xyxy[0]
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, box)
            cv2.rectangle(depth_colored, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)
            label_text = f"{yolo_model.names[int(result.cls[0])]}: {result.conf[0]:.2f}"
            cv2.putText(depth_colored, label_text, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return depth_colored

def initialize_video_writer(save_video=False, output_dir="visualization_output", fps=6, resolution=(1280, 960)):
    if save_video:
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(f"{output_dir}/output.mp4", fourcc, fps, resolution)
    return None

def get_object_mask(frame, prompt):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    seg_inputs = seg_processor(text=[prompt], images=[pil_image], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        seg_outputs = seg_model(**seg_inputs)
    mask = torch.sigmoid(seg_outputs.logits.unsqueeze(1)[0][0]).cpu().numpy()
    binary_mask = (cv2.resize(mask, (frame.shape[1], frame.shape[0])) > 0.5).astype(np.uint8) * 255
    return binary_mask