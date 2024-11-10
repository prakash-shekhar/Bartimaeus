#library imports
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import time
import numpy as np
import cv2
from ultralytics import YOLO
import PathFinder
import sounddevice as sd
import torch
import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# file imports
from utils import *
from whisper import get_target_object

device = "mps"
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
model = YOLO('yolov5su.pt')

RECORD_DURATION = 5

if __name__ == "__main__":
    target_obj = get_target_object(record_duration=RECORD_DURATION)
    speak_mac(f"Perfect! Let's go find your {target_obj}")


