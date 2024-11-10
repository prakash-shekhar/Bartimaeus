import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-medium", torch_dtype=torch.float32, device="mps")
sample_rate = 16000

def record_audio(record_duration):
    print("Recording... Please speak.")
    audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, device =1, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def extract_object(text):
    match = re.search(r'help me find my (.+)', text.lower())
    if match:
        return match.group(1).strip()
    return None

def transcribe_audio(audio_data):
    print("Transcribing audio...")
    transcription = whisper({"array": audio_data, "sampling_rate": sample_rate, "language": "en"})
    transcribed_text = transcription["text"]
    print("You said:", transcribed_text)
    return extract_object(transcribed_text)

def get_target_object(record_duration):
    audio_data = record_audio(record_duration)
    object_name = transcribe_audio(audio_data)
    if object_name:
        print("Object to detect:", object_name)
    else:
        print("No object found in the transcription.")
    return object_name