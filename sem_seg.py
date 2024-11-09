import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import pipeline
from huggingface_hub import hf_hub_download
import time

model_type = "vit_b"
sam_checkpoint = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
sam = sam_model_registry[model_type](sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
depth_model_checkpoint = "depth-anything/Depth-Anything-V2-Small-hf"
depth_pipe = pipeline("depth-estimation", model=depth_model_checkpoint, device=device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb_frame)
    pil_image = Image.fromarray(rgb_frame)
    depth = depth_pipe(pil_image)["depth"]
    depth_array = np.array(depth)
    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

    walkable_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for mask in masks:
        mask_area = mask['segmentation']
        segment_depth = depth_array[mask_area].mean()
        if segment_depth > 100:
            walkable_mask[mask_area] = 255
        else:
            walkable_mask[mask_area] = 0

    walkable_color_map = cv2.applyColorMap(walkable_mask, cv2.COLORMAP_WINTER)
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Original', frame)
    cv2.imshow('Depth Map (Color)', depth_colored)
    cv2.imshow('Walkable Map (Color)', walkable_color_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from PIL import Image
# import torch
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# from transformers import pipeline
# import time

# # Load SAM model for segmentation
# model_type = "vit_b"  # Choose a smaller model for efficiency
# sam_checkpoint = "path/to/sam/checkpoint"  # Update with your SAM model checkpoint path
# sam = sam_model_registry[model_type](sam_checkpoint)
# mask_generator = SamAutomaticMaskGenerator(sam)

# # Load depth estimation pipeline
# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# depth_model_checkpoint = "depth-anything/Depth-Anything-V2-Small-hf"
# depth_pipe = pipeline("depth-estimation", model=depth_model_checkpoint, device=device)

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Main loop for real-time processing
# while True:
#     start_time = time.time()
    
#     # Capture a frame from the camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert frame to PIL format for SAM and depth models
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Generate segmentation masks using SAM
#     masks = mask_generator.generate(pil_image)
    
#     # Perform depth estimation
#     depth = depth_pipe(pil_image)["depth"]
#     depth_array = np.array(depth)
#     depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

#     # Initialize the walkability mask (binary mask)
#     walkable_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
#     # Iterate through SAM masks to classify walkable and non-walkable regions
#     for mask in masks:
#         mask_area = mask['segmentation']
#         # Calculate average depth of this segment
#         segment_depth = depth_array[mask_area].mean()

#         # Define a threshold to classify areas as walkable based on depth and position
#         if segment_depth > 100:  # Arbitrary threshold to exclude far objects (adjust as needed)
#             walkable_mask[mask_area] = 255  # Mark as walkable (white)
#         else:
#             walkable_mask[mask_area] = 0     # Mark as non-walkable (black)

#     # Convert walkable mask to color for visualization
#     walkable_color_map = cv2.applyColorMap(walkable_mask, cv2.COLORMAP_WINTER)
    
#     # Calculate FPS
#     fps = 1 / (time.time() - start_time)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # Display the original frame, depth map, and walkability mask
#     cv2.imshow('Original', frame)
#     cv2.imshow('Depth Map (Color)', depth_colored)
#     cv2.imshow('Walkable Map (Color)', walkable_color_map)

#     # Break the loop on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()