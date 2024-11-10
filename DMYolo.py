from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from utils import *
depth_colored = None
import PathFinder
import time


AREA_THRESHOLD = 2.2
DEPTH_THRESHOLD = 25
TIME_SPEAK_INTERVAL = 1

DEPTH_ADJUST_COUNTER = 1

device = "mps"
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

target_obj = "bottle"
obstacle_obj =  ["chair", "backpack"]
# obstacle_obj_2 =  "backpack"
speak_mac(f"Perfect! Let's go find your '{target_obj}'")

color_map = {
    "backpack": (255, 0, 0),
    "chair": (0, 255, 0),
    "bottle": (0, 0, 255),
    "cup": (0, 255, 255)
}
initial_depth = {}
initial_area = {}

move_arr=[]
start = time.time()
interval_time = 0

while True:
    data = []
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # if DEPTH_ADJUST_COUNTER < 1:
    #     speak_mac(f"The {target_obj} has been found!")
    #     DEPTH_ADJUST_COUNTER = 1

    frame = cv2.resize(frame, (640, 480))
    combined_mask = np.zeros_like(frame)

    target_mask = get_object_mask(frame, prompt=target_obj)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if target_contours:
        largest_contour = max(target_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = cv2.contourArea(largest_contour)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth = depth_pipe(image)["depth"]
        depth_array = np.array(depth)

        target_depth_values = depth_array[y:y+h, x:x+w]
        average_depth = np.mean(target_depth_values)

        depth_colored = cv2.applyColorMap(cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
        box_color = color_map.get(target_obj, (255, 255, 255))
        cv2.rectangle(depth_colored, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(depth_colored, f"{target_obj}: Avg Depth {average_depth:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        target_overlay = np.zeros_like(frame)
        target_overlay[target_mask == 255] = box_color
        combined_mask = cv2.addWeighted(combined_mask, 1, target_overlay, 1, 0)

        data.append((target_obj, x, y, x+w, y+h, average_depth))

    if obstacle_obj is not None:
        obstacle_mask = get_object_mask(frame, prompt=obstacle_obj)
        obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if obstacle_contours:
            largest_obstacle_contour = max(obstacle_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_obstacle_contour)
            obstacle_depth_values = depth_array[y:y+h, x:x+w]
            obstacle_average_depth = np.mean(obstacle_depth_values)

            obstacle_color = color_map.get(obstacle_obj, (255, 255, 255))
            cv2.rectangle(depth_colored, (x, y), (x + w, y + h), obstacle_color, 2)
            cv2.putText(depth_colored, f"{obstacle_obj}: Avg Depth {obstacle_average_depth:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obstacle_color, 1)
            obstacle_overlay = np.zeros_like(frame)
            obstacle_overlay[obstacle_mask == 255] = obstacle_color
            combined_mask = cv2.addWeighted(combined_mask, 1, obstacle_overlay, 1, 0)
            data.append((obstacle_obj, x , y, x+w , y+h , average_depth))

    # if depth_colored is None:
    #     if DEPTH_ADJUST_COUNTER==0:
    #         speak_mac(f"The {target_obj} does not exist in your surroundings!")
    #         break
    #     DEPTH_ADJUST_COUNTER-=1
    #     speak_mac(f"The {target_obj} cannot be seen. Please turn around!")
    #     time.sleep(1.5)
    #     continue

    if depth_colored is not None:
        # print(data)
        seg_frame = run_semantic_segmentation(frame, prompt="ground", overlay_original=True)
        cv2.imshow('Original frame', frame)
        cv2.imshow("Original with Semantic Segmentation", seg_frame)
        cv2.imshow('Depth Map (Color)', depth_colored)
        overlay = cv2.addWeighted(frame, 0.6, combined_mask, 0.4, 0)
        cv2.imshow('Original with Ground Segmentation Overlay', overlay)
        # top_row = np.hstack((frame, seg_frame))
        # bottom_row = np.hstack((depth_colored, overlay))
        # combined_display = np.vstack((top_row, bottom_row))
        # cv2.imshow('Combined Display', combined_display)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    move, location, bool_et = PathFinder.next_instruction(data, target_obj)

    if target_obj not in initial_depth:
        initial_depth[target_obj] = average_depth
        initial_area[target_obj] = area
    else:
        if area >= AREA_THRESHOLD * initial_area[target_obj] and average_depth >= initial_depth[target_obj] + DEPTH_THRESHOLD and location=="middle of screen":
            speak_mac(f"You are now at the {target_obj}. Please use your hands to slowly get a feeling of your environment!")
            print(f"You are now at the {target_obj}.")
            break

    interval_time += time.time() - start_time
    start_time = time.time()

    if(interval_time > TIME_SPEAK_INTERVAL):
        if (bool_et): time.sleep(0.5)
        speak_mac(move)
        interval_time = 0
        i += 1


cap.release()
cv2.destroyAllWindows()