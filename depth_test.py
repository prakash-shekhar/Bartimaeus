#New Code (0.5 Res): 
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import time
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

i = 0
while True:
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame to half its original resolution
    frame_small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Convert the resized frame to RGB format for the pipeline
    image = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    depth = pipe(image)["depth"]
    depth_array = np.array(depth)
    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save only the smaller depth image
    if i < 10:
        cv2.imwrite(f"depth_photos/0{i}.png", depth_colored)
        i += 1

    # Display the original frame and depth map in their original size
    cv2.imshow('Original', frame)
    cv2.imshow('Depth Map (Color)', depth_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








#Old code with 1920x1080 Res.
# from transformers import pipeline
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import torch
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# i = 0
# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     depth = pipe(image)["depth"]
#     depth_array = np.array(depth)
#     depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

#     fps = 1 / (time.time() - start_time)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     if i < 10: 
#         cv2.imwrite(f"depth_photos/0{i}.png", depth_colored)
#         i+=1

#     cv2.imshow('Original', frame)
#     cv2.imshow('Depth Map (Color)', depth_colored)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()













# from transformers import pipeline
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import torch

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
# pipe = pipeline("depth-estimation", model=checkpoint, device=device)

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     depth = pipe(image)["depth"]
#     depth_array = np.array(depth)
#     depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

#     fps = 1 / (time.time() - start_time)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     cv2.imshow('Original', frame)
#     cv2.imshow('Depth Map (Color)', depth_colored)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()